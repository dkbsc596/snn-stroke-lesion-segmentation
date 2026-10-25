"""
Leave-One-Patient-Out (LOPO) 2.5D Spiking U-Net for ISLES2022  [2.5D = k adjacent axial slices stacked as channels, 2D conv]
- 30 epochs per fold
- Resumable: finished folds are skipped, interrupted folds resume from last epoch
- Pure PyTorch LIF neurons (no snntorch needed)

Run (background, survives PuTTY close):  bash run_lopo.sh
"""
import os, glob, csv, time, math, argparse, random
import numpy as np
import nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

p = argparse.ArgumentParser()
p.add_argument("--data_root", default="/home/krishna_19622/dataset/isles_archive/ISLES-2022_extracted/ISLES-2022")
p.add_argument("--out_dir", default="/home/krishna_19622/dataset/results_lopo_snn_2.5d")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--T", type=int, default=4, help="SNN time steps")
p.add_argument("--k", type=int, default=3, help="neighbouring slices (2.5D), odd")
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--base", type=int, default=32)
p.add_argument("--start_fold", type=int, default=0)
p.add_argument("--end_fold", type=int, default=-1, help="-1 = all patients")
p.add_argument("--neg_ratio", type=float, default=1.0, help="background slices per lesion slice each epoch")
p.add_argument("--track_test", type=int, default=1, help="1 = also log held-out Dice each epoch (monitoring only, never used to pick a model)")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out_dir, exist_ok=True)
LOG = os.path.join(args.out_dir, "run.log")
RES = os.path.join(args.out_dir, "lopo_results.csv")


def log(*a):
    s = time.strftime("[%Y-%m-%d %H:%M:%S] ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


# ---------------------------------------------------------------- data
def find_cases(root):
    """ISLES2022 layout: rawdata/sub-X/ses-0001/{dwi,adc}/*.nii.gz ; derivatives/sub-X/ses-0001/*msk.nii.gz
    Edit the globs below if your prepared folder differs."""
    cases = []
    for dwi in sorted(glob.glob(os.path.join(root, "sub-*", "ses-*", "dwi", "*_dwi.nii.gz"))):
        sub = dwi.split(os.sep)
        sid = [s for s in sub if s.startswith("sub-")][0]
        adc = dwi.replace("_dwi.nii.gz", "_adc.nii.gz").replace("/dwi/", "/dwi/")
        msk = glob.glob(os.path.join(root, "derivatives", sid, "ses-*", "*_msk.nii.gz"))
        if os.path.exists(adc) and msk:
            cases.append((sid, dwi, adc, msk[0]))
    return cases


def zscore(v):
    m = v > 0
    if m.sum() < 10:
        return np.zeros_like(v, dtype=np.float32)
    mu, sd = v[m].mean(), v[m].std() + 1e-6
    out = (v - mu) / sd
    out[~m] = 0
    return out.astype(np.float32)


TARGET_MM = 2.0  # in-plane spacing all cases are resampled to


def resample_xy(vol, sx, sy):
    """in-plane (X,Y) bilinear resize of a (X,Y,Z) volume by factors sx, sy"""
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).permute(2, 0, 1)[:, None]
    size = (max(1, round(t.shape[2] * sx)), max(1, round(t.shape[3] * sy)))
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t[:, 0].permute(1, 2, 0).numpy()


def load_case(c):
    _, dwi, adc, msk = c
    im = nib.load(dwi)
    zx, zy = [float(z) for z in im.header.get_zooms()[:2]]
    d = im.get_fdata(dtype=np.float32)
    a = nib.load(adc).get_fdata(dtype=np.float32)
    m = (nib.load(msk).get_fdata() > 0.5).astype(np.float32)
    sx, sy = zx / TARGET_MM, zy / TARGET_MM
    if abs(sx - 1) > 0.03 or abs(sy - 1) > 0.03:
        d, a, m = resample_xy(d, sx, sy), resample_xy(a, sx, sy), resample_xy(m, sx, sy)
    d, a = zscore(d), zscore(a)
    return np.stack([d, a], 0), (m > 0.5).astype(np.uint8)  # (2,X,Y,Z), (X,Y,Z)


SZ = 128


def pad_crop(x):  # x: (C,H,W) -> (C,SZ,SZ)
    _, h, w = x.shape
    if h > SZ:
        s = (h - SZ) // 2; x = x[:, s:s + SZ]
    if w > SZ:
        s = (w - SZ) // 2; x = x[:, :, s:s + SZ]
    _, h, w = x.shape
    ph, pw = SZ - h, SZ - w
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


class SliceDS(Dataset):
    def __init__(self, vols, k, train):
        self.vols, self.k, self.train = vols, k, train
        self.pos, self.neg = [], []
        for vi, (_, m) in enumerate(vols):
            for z in range(m.shape[2]):
                (self.pos if m[:, :, z].any() else self.neg).append((vi, z))
        self.items = self.pos + self.neg

    def resample(self, ratio):
        n = int(len(self.pos) * ratio)
        self.items = self.pos + random.sample(self.neg, min(n, len(self.neg)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vi, z = self.items[i]
        img, m = self.vols[vi]
        Z, h = img.shape[3], self.k // 2
        idx = [min(max(z + o, 0), Z - 1) for o in range(-h, h + 1)]
        x = torch.from_numpy(img[:, :, :, idx]).permute(0, 3, 1, 2).reshape(-1, img.shape[1], img.shape[2])
        y = torch.from_numpy(m[:, :, z].astype(np.float32))[None]
        x, y = pad_crop(x), pad_crop(y)
        if self.train:
            if random.random() < 0.5: x, y = x.flip(2), y.flip(2)
            if random.random() < 0.5: x, y = x.flip(1), y.flip(1)
        return x, y


# ---------------------------------------------------------------- SNN
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + (math.pi * x) ** 2)  # arctan surrogate


class LIF(nn.Module):
    def __init__(self, beta=0.5, thr=1.0):
        super().__init__()
        self.beta, self.thr, self.mem = beta, thr, None

    def reset(self):
        self.mem = None

    def forward(self, x):
        self.mem = x if self.mem is None else self.beta * self.mem + x
        s = SpikeFn.apply(self.mem - self.thr)
        self.mem = self.mem - s.detach() * self.thr  # soft reset
        return s


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), LIF(),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), LIF())


class SpikingUNet(nn.Module):
    def __init__(self, cin, base=32, T=4):
        super().__init__()
        b = base
        self.T = T
        self.e1, self.e2, self.e3, self.bt = block(cin, b), block(b, 2 * b), block(2 * b, 4 * b), block(4 * b, 8 * b)
        self.d3, self.d2, self.d1 = block(12 * b, 4 * b), block(6 * b, 2 * b), block(3 * b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def reset(self):
        for m in self.modules():
            if isinstance(m, LIF): m.reset()

    def step(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bt(F.max_pool2d(e3, 2))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        d3 = self.d3(torch.cat([up(b), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.out(d1)

    def forward(self, x):
        self.reset()
        return torch.stack([self.step(x) for _ in range(self.T)]).mean(0)  # direct encoding


def loss_fn(lg, y):
    bce = F.binary_cross_entropy_with_logits(lg, y)
    p = torch.sigmoid(lg)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


# ---------------------------------------------------------------- LOPO
@torch.no_grad()
def eval_volume(model, vol, k):
    model.eval()
    img, m = vol
    ds = SliceDS([vol], k, False)
    ds.items = [(0, z) for z in range(m.shape[2])]
    pred = np.zeros((SZ, SZ, m.shape[2]), np.uint8)
    gt = np.zeros_like(pred)
    for i in range(0, len(ds), 16):
        xs, ys = zip(*[ds[j] for j in range(i, min(i + 16, len(ds)))])
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            pr = (torch.sigmoid(model(torch.stack(xs).to(dev))) > 0.5).cpu().numpy()[:, 0]
        for j in range(len(xs)):
            pred[:, :, i + j] = pr[j]
            gt[:, :, i + j] = ys[j][0].numpy() > 0.5
    inter = (pred & gt).sum()
    s = pred.sum() + gt.sum()
    dice = 1.0 if s == 0 else 2 * inter / s
    return float(dice), int(gt.sum()), int(pred.sum())


def done_ids():
    if not os.path.exists(RES): return set()
    with open(RES) as f: return {r["patient"] for r in csv.DictReader(f)}


def main():
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cases = find_cases(args.data_root)
    log(f"Found {len(cases)} cases | device={dev} | epochs={args.epochs} T={args.T} k={args.k}")
    if not cases:
        raise SystemExit("No cases found - fix --data_root / globs in find_cases()")
    log("Loading all volumes into RAM ...")
    vols = [load_case(c) for c in cases]
    n = len(cases)
    end = n if args.end_fold < 0 else min(args.end_fold, n)

    for f in range(args.start_fold, end):
        pid = cases[f][0]
        if pid in done_ids():
            log(f"Fold {f}/{n} {pid} already done - skip"); continue
        fdir = os.path.join(args.out_dir, f"fold_{f:03d}_{pid}")
        os.makedirs(fdir, exist_ok=True)
        ck = os.path.join(fdir, "last.pt")

        tr = SliceDS([vols[i] for i in range(n) if i != f], args.k, True)
        model = SpikingUNet(2 * args.k, args.base, args.T).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
        ep0 = 0
        if os.path.exists(ck):
            s = torch.load(ck, map_location=dev)
            model.load_state_dict(s["m"]); opt.load_state_dict(s["o"]); sched.load_state_dict(s["s"])
            ep0 = s["ep"]
            log(f"Fold {f} {pid}: resumed from epoch {ep0}")

        for ep in range(ep0, args.epochs):
            model.train(); tr.resample(args.neg_ratio)
            dl = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
            tot, tdice, t0 = 0.0, 0.0, time.time()
            for x, y in dl:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    out = model(x).float()
                    loss = loss_fn(out, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += loss.item()
                with torch.no_grad():
                    pb = (torch.sigmoid(out) > 0.5).float()
                    tdice += ((2 * (pb * y).sum() + 1) / (pb.sum() + y.sum() + 1)).item()
            sched.step()
            torch.save({"m": model.state_dict(), "o": opt.state_dict(), "s": sched.state_dict(), "ep": ep + 1}, ck)
            nb = max(len(dl), 1)
            tdc = tdice / nb
            test_d = ""
            if args.track_test:
                test_d = f"{eval_volume(model, vols[f], args.k)[0]:.4f}"
            lr_now = opt.param_groups[0]["lr"]
            hp = os.path.join(args.out_dir, "training_history.csv")
            newh = not os.path.exists(hp)
            with open(hp, "a", newline="") as hf:
                hw = csv.writer(hf)
                if newh: hw.writerow(["fold", "patient", "epoch", "train_loss", "train_dice", "heldout_dice_monitor", "lr", "epoch_sec"])
                hw.writerow([f, pid, ep + 1, f"{tot / nb:.4f}", f"{tdc:.4f}", test_d, f"{lr_now:.6f}", f"{time.time() - t0:.0f}"])
            log(f"Fold {f}/{n} {pid} | epoch {ep + 1}/{args.epochs} | loss {tot / nb:.4f} | train Dice {tdc:.4f} | held-out Dice {test_d} | lr {lr_now:.5f} | {time.time() - t0:.0f}s")

        dice, gts, prs = eval_volume(model, vols[f], args.k)
        log(f"Fold {f} {pid} TEST Dice = {dice:.4f}")
        new = not os.path.exists(RES)
        with open(RES, "a", newline="") as fh:
            w = csv.writer(fh)
            if new: w.writerow(["fold", "patient", "dice", "gt_voxels", "pred_voxels"])
            w.writerow([f, pid, f"{dice:.4f}", gts, prs])
        torch.save(model.state_dict(), os.path.join(fdir, "final.pt"))
        os.remove(ck)

    if os.path.exists(RES):
        d = [float(r["dice"]) for r in csv.DictReader(open(RES))]
        log(f"LOPO finished folds={len(d)} mean Dice={np.mean(d):.4f} +/- {np.std(d):.4f}")
    open(os.path.join(args.out_dir, "ALL_DONE" if end == n else "PART_DONE"), "w").close()


if __name__ == "__main__":
    main()
