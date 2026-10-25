"""
Unified experiment framework - ISLES2022, 2.5D U-Net (spiking or ANN)

One run = one configuration, evaluated patient-wise with the SAME folds for every config:
    python3 exp_framework.py --model snn --modality dwi_adc --T 2 --k 3        # main model
    python3 exp_framework.py --model snn --modality dwi     --T 2 --k 3        # input ablation
    python3 exp_framework.py --model ann --modality dwi_adc --k 3              # ANN baseline
    python3 exp_framework.py --folds 0 ...                                     # LOPO (leave-one-patient-out)

Per run it writes (out_root/<run name>/):
    per_patient.csv     Dice, HD95, precision, recall, volumes for every patient (one row per patient)
    spike_stats.csv     per patient: slices, dense MACs, synaptic ops (SOPs)  -> energy estimate
    layer_rates.csv     per patient & LIF layer: spikes, neuron-steps         -> firing-rate analysis
    training_history.csv  loss / train Dice / lr per epoch per fold
    qual/<patient>.npz  DWI, ADC, GT, prediction of the largest-lesion slice  -> qualitative figure
    fold_<f>.pt, ALL_DONE
Resumable: finished folds are skipped, an interrupted fold continues from its last epoch.
"""
import os, glob, csv, time, math, json, argparse, random
import numpy as np
import nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

p = argparse.ArgumentParser()
p.add_argument("--data_root", default="/home/krishna_19622/dataset/isles_archive/ISLES-2022_extracted/ISLES-2022")
p.add_argument("--out_root", default="/home/krishna_19622/dataset/experiments")
p.add_argument("--model", choices=["snn", "ann"], default="snn")
p.add_argument("--modality", choices=["dwi", "dwi_adc"], default="dwi_adc")
p.add_argument("--T", type=int, default=2, help="SNN time steps (ignored for ann)")
p.add_argument("--k", type=int, default=3, help="slices per sample (odd). k=1 is plain 2D")
p.add_argument("--folds", type=int, default=5, help="number of folds; 0 = leave-one-patient-out")
p.add_argument("--only_fold", type=int, default=-1)
p.add_argument("--import_lopo", default="", help="folder of an ALREADY FINISHED lopo_snn_2.5d.py run: evaluate its saved fold models, no retraining")
p.add_argument("--lopo_subset", type=int, default=0, help="LOPO on N patients only (spread over lesion sizes); 0 = all patients")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--base", type=int, default=32)
p.add_argument("--neg_ratio", type=float, default=0.5)
p.add_argument("--min_component", type=int, default=0, help="post-processing: drop 3D connected components smaller than N voxels (the largest one is always kept); 0 = off")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
assert args.k % 2 == 1, "k must be odd"
if args.model == "ann":
    args.T = 1

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAME = f"{args.model}_{args.modality}_T{args.T}_k{args.k}_s{args.seed}" + ("_lopo" if args.folds == 0 else f"_cv{args.folds}")
if args.min_component > 0:
    NAME += f"_pp{args.min_component}"
RUN = os.path.join(args.out_root, NAME)
os.makedirs(os.path.join(RUN, "qual"), exist_ok=True)
json.dump(vars(args), open(os.path.join(RUN, "config.json"), "w"), indent=1)
LOG = os.path.join(RUN, "run.log")
SZ, TARGET_MM = 128, 2.0
E_MAC, E_AC = 4.6e-12, 0.9e-12  # J per MAC / per accumulate (45 nm, Horowitz 2014) - assumption, see paper text


def log(*a):
    s = time.strftime("[%Y-%m-%d %H:%M:%S] ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    open(LOG, "a").write(s + "\n")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ data
def find_cases(root):
    cases = []
    for dwi in sorted(glob.glob(os.path.join(root, "sub-*", "ses-*", "dwi", "*_dwi.nii.gz"))):
        sid = [s for s in dwi.split(os.sep) if s.startswith("sub-")][0]
        adc = dwi.replace("_dwi.nii.gz", "_adc.nii.gz")
        msk = glob.glob(os.path.join(root, "derivatives", sid, "ses-*", "*_msk.nii.gz"))
        if os.path.exists(adc) and msk:
            cases.append((sid, dwi, adc, msk[0]))
    return cases


def zscore(v):
    m = v > 0
    if m.sum() < 10:
        return np.zeros_like(v, dtype=np.float32)
    out = (v - v[m].mean()) / (v[m].std() + 1e-6)
    out[~m] = 0
    return out.astype(np.float32)


def resample_xy(vol, sx, sy):
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).permute(2, 0, 1)[:, None]
    size = (max(1, round(t.shape[2] * sx)), max(1, round(t.shape[3] * sy)))
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t[:, 0].permute(1, 2, 0).numpy()


def load_case(c, use_adc):
    _, dwi, adc, msk = c
    im = nib.load(dwi)
    zx, zy, zz = [float(z) for z in im.header.get_zooms()[:3]]
    d = im.get_fdata(dtype=np.float32)
    a = nib.load(adc).get_fdata(dtype=np.float32) if use_adc else None
    m = (nib.load(msk).get_fdata() > 0.5).astype(np.float32)
    sx, sy = zx / TARGET_MM, zy / TARGET_MM
    if abs(sx - 1) > 0.03 or abs(sy - 1) > 0.03:
        d, m = resample_xy(d, sx, sy), resample_xy(m, sx, sy)
        if a is not None:
            a = resample_xy(a, sx, sy)
    ch = [zscore(d)] + ([zscore(a)] if use_adc else [])
    return np.stack(ch, 0), (m > 0.5).astype(np.uint8), zz


def pad_crop(x):
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
        for vi, v in enumerate(vols):
            m = v[1]
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
        img, m = self.vols[vi][0], self.vols[vi][1]
        Z, h = img.shape[3], self.k // 2
        idx = [min(max(z + o, 0), Z - 1) for o in range(-h, h + 1)]
        x = torch.from_numpy(img[:, :, :, idx]).permute(0, 3, 1, 2).reshape(-1, img.shape[1], img.shape[2])  # channel = c*k + slice
        y = torch.from_numpy(m[:, :, z].astype(np.float32))[None]
        x, y = pad_crop(x), pad_crop(y)
        if self.train:
            if random.random() < 0.5: x, y = x.flip(2), y.flip(2)
            if random.random() < 0.5: x, y = x.flip(1), y.flip(1)
        return x, y


# ------------------------------------------------------------------ models
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + (math.pi * x) ** 2)


class LIF(nn.Module):
    def __init__(self, beta=0.5, thr=1.0):
        super().__init__()
        self.beta, self.thr, self.mem = beta, thr, None

    def reset(self):
        self.mem = None

    def forward(self, x):
        self.mem = x if self.mem is None else self.beta * self.mem + x
        s = SpikeFn.apply(self.mem - self.thr)
        self.mem = self.mem - s.detach() * self.thr
        return s


class UNet(nn.Module):
    """Same topology for SNN (LIF) and ANN (ReLU) -> fair comparison."""
    def __init__(self, cin, base=32, T=2, spiking=True):
        super().__init__()
        self.T, self.spiking = T, spiking
        act = (lambda: LIF()) if spiking else (lambda: nn.ReLU())
        b = base

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), act(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), act())
        self.e1, self.e2, self.e3, self.bt = blk(cin, b), blk(b, 2 * b), blk(2 * b, 4 * b), blk(4 * b, 8 * b)
        self.d3, self.d2, self.d1 = blk(12 * b, 4 * b), blk(6 * b, 2 * b), blk(3 * b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def reset(self):
        for m in self.modules():
            if isinstance(m, LIF):
                m.reset()

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
        if not self.spiking:
            return self.step(x)
        self.reset()
        return torch.stack([self.step(x) for _ in range(self.T)]).mean(0)


def loss_fn(lg, y):
    bce = F.binary_cross_entropy_with_logits(lg, y)
    pr = torch.sigmoid(lg)
    return bce + 1 - (2 * (pr * y).sum() + 1) / (pr.sum() + y.sum() + 1)


# ------------------------------------------------------------------ spike / op statistics
class Stats:
    """Counts firing rates (LIF outputs) and operations (first conv = dense MACs, all other convs in an
    SNN receive binary spikes -> synaptic operations SOPs = input spikes x fan-out)."""
    def __init__(self, model):
        self.model, self.h = model, []
        self.reset()

    def reset(self):
        self.spk, self.cnt, self.macs, self.sops = {}, {}, 0.0, 0.0

    def attach(self):
        for name, m in self.model.named_modules():
            if isinstance(m, LIF):
                self.h.append(m.register_forward_hook(self._lif(name)))
            elif isinstance(m, nn.Conv2d):
                self.h.append(m.register_forward_hook(self._conv(name)))

    def detach(self):
        for h in self.h:
            h.remove()
        self.h = []

    def _lif(self, name):
        def hook(mod, inp, out):
            self.spk[name] = self.spk.get(name, 0.0) + float(out.float().sum())
            self.cnt[name] = self.cnt.get(name, 0) + out.numel()
        return hook

    def _conv(self, name):
        def hook(mod, inp, out):
            kk = mod.kernel_size[0] * mod.kernel_size[1]
            if (not self.model.spiking) or name == "e1.0":
                self.macs += out.numel() * mod.in_channels * kk
            else:
                self.sops += float(inp[0].float().sum()) * mod.out_channels * kk
        return hook


# ------------------------------------------------------------------ metrics
def hd95(pred, gt, sp):
    if ndi is None or pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    sp_, sg_ = pred & ~ndi.binary_erosion(pred), gt & ~ndi.binary_erosion(gt)
    dg = ndi.distance_transform_edt(~sg_, sampling=sp)
    dp = ndi.distance_transform_edt(~sp_, sampling=sp)
    return float(np.percentile(np.concatenate([dg[sp_], dp[sg_]]), 95))


@torch.no_grad()
def eval_volume(model, vol, k, st, pid):
    model.eval()
    img, m, zz = vol
    Z, h, C = m.shape[2], k // 2, img.shape[0]
    ds = SliceDS([vol], k, False)
    ds.items = [(0, z) for z in range(Z)]
    pred = np.zeros((SZ, SZ, Z), bool)
    gt = np.zeros_like(pred)
    dwi_c = np.zeros((Z, SZ, SZ), np.float16)
    adc_c = np.zeros((Z, SZ, SZ), np.float16)
    st.reset(); st.attach()
    for i in range(0, Z, 16):
        batch = [ds[j] for j in range(i, min(i + 16, Z))]
        x = torch.stack([b[0] for b in batch]).to(dev)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            pr = (torch.sigmoid(model(x).float()) > 0.5).cpu().numpy()[:, 0]
        for j, b in enumerate(batch):
            pred[:, :, i + j] = pr[j]
            gt[:, :, i + j] = b[1][0].numpy() > 0.5
            dwi_c[i + j] = b[0][h].numpy()
            if C == 2:
                adc_c[i + j] = b[0][k + h].numpy()
    st.detach()
    if args.min_component > 0 and ndi is not None and pred.any():
        lab, nl = ndi.label(pred)
        if nl > 1:
            sz = np.asarray(ndi.sum(pred, lab, range(1, nl + 1)))
            keep = [i + 1 for i, v in enumerate(sz) if v >= args.min_component or v == sz.max()]
            pred = np.isin(lab, keep)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    dice = 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    zs = int(np.argmax(gt.sum((0, 1))))
    np.savez_compressed(os.path.join(RUN, "qual", f"{pid}.npz"), dwi=dwi_c[zs], adc=adc_c[zs],
                        gt=gt[:, :, zs], pred=pred[:, :, zs], dice=dice, slice=zs)
    return dict(dice=dice, hd95=hd95(pred, gt, (TARGET_MM, TARGET_MM, zz)), precision=prec, recall=rec,
                gt_vox=int(gt.sum()), pred_vox=int(pred.sum()), n_slices=Z, macs=st.macs, sops=st.sops,
                spk=dict(st.spk), cnt=dict(st.cnt))


def append_rows(fname, header, rows):
    path = os.path.join(RUN, fname)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ main
def main():
    seed_all(args.seed)
    use_adc = args.modality == "dwi_adc"
    cases = find_cases(args.data_root)
    log(f"RUN {NAME} | {len(cases)} cases | device={dev} | folds={args.folds or 'LOPO'}")
    if not cases:
        raise SystemExit("No cases found - check --data_root")
    vols = [load_case(c, use_adc) for c in cases]
    n = len(cases)
    nf = n if args.folds == 0 else args.folds
    # identical, size-balanced patient-wise folds for every configuration (depends only on the data)
    sizes = np.array([v[1].sum() for v in vols])
    fold_of = np.zeros(n, int)
    if args.folds == 0:
        fold_of = np.arange(n)  # LOPO: fold f = patient f (same numbering as lopo_snn_2.5d.py)
    else:
        for r, i in enumerate(np.argsort(sizes, kind="stable")):
            fold_of[i] = r % nf
    run_folds = list(range(nf))
    if args.folds == 0 and args.lopo_subset > 0:
        o_ = np.argsort(sizes, kind="stable")
        run_folds = sorted({int(o_[i]) for i in np.linspace(0, n - 1, args.lopo_subset).astype(int)})
        log(f"LOPO on a subset of {len(run_folds)} patients spread over lesion sizes")
    cin = (2 if use_adc else 1) * args.k

    for f in run_folds:
        if args.only_fold >= 0 and f != args.only_fold:
            continue
        done = os.path.join(RUN, f"fold_{f}.done")
        if os.path.exists(done):
            log(f"fold {f} done - skip"); continue
        seed_all(args.seed + f)
        tr_idx = [i for i in range(n) if fold_of[i] != f]
        te_idx = [i for i in range(n) if fold_of[i] == f]
        tr = None if args.import_lopo else SliceDS([vols[i] for i in tr_idx], args.k, True)
        model = UNet(cin, args.base, args.T, spiking=args.model == "snn").to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        ck, ep0 = os.path.join(RUN, f"fold_{f}_last.pt"), 0
        if args.import_lopo:
            pth = os.path.join(args.import_lopo, f"fold_{f:03d}_{cases[f][0]}", "final.pt")
            model.load_state_dict(torch.load(pth, map_location=dev))
            ep0 = args.epochs  # no training, evaluation only
        elif os.path.exists(ck):
            s = torch.load(ck, map_location=dev)
            model.load_state_dict(s["m"]); opt.load_state_dict(s["o"]); sched.load_state_dict(s["s"]); ep0 = s["ep"]
            log(f"fold {f}: resumed at epoch {ep0}")
        nparams = sum(q.numel() for q in model.parameters() if q.requires_grad)
        log(f"fold {f + 1}/{nf} | train {len(tr_idx)} patients, test {len(te_idx)} | params {nparams}")

        for ep in range(ep0, args.epochs):
            model.train(); tr.resample(args.neg_ratio)
            dl = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
            tot, td, t0 = 0.0, 0.0, time.time()
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
                    td += ((2 * (pb * y).sum() + 1) / (pb.sum() + y.sum() + 1)).item()
            sched.step()
            nb = max(len(dl), 1)
            torch.save({"m": model.state_dict(), "o": opt.state_dict(), "s": sched.state_dict(), "ep": ep + 1}, ck)
            append_rows("training_history.csv", ["fold", "epoch", "train_loss", "train_dice", "lr", "sec"],
                        [[f, ep + 1, f"{tot / nb:.4f}", f"{td / nb:.4f}", f"{opt.param_groups[0]['lr']:.6f}", f"{time.time() - t0:.0f}"]])
            log(f"fold {f + 1}/{nf} | epoch {ep + 1}/{args.epochs} | loss {tot / nb:.4f} | train Dice {td / nb:.4f} | {time.time() - t0:.0f}s")

        st = Stats(model)
        pp, ss, lr_rows = [], [], []
        for i in te_idx:
            pid = cases[i][0]
            r = eval_volume(model, vols[i], args.k, st, pid)
            pp.append([pid, f, f"{r['dice']:.4f}", f"{r['hd95']:.3f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}", r["gt_vox"], r["pred_vox"]])
            ss.append([pid, r["n_slices"], f"{r['macs']:.0f}", f"{r['sops']:.0f}"])
            for ln in r["spk"]:
                lr_rows.append([pid, ln, f"{r['spk'][ln]:.0f}", r["cnt"][ln]])
        append_rows("per_patient.csv", ["patient", "fold", "dice", "hd95", "precision", "recall", "gt_vox", "pred_vox"], pp)
        append_rows("spike_stats.csv", ["patient", "n_slices", "macs", "sops"], ss)
        if lr_rows:
            append_rows("layer_rates.csv", ["patient", "layer", "spikes", "neuron_steps"], lr_rows)
        if not args.import_lopo:
            torch.save(model.state_dict(), os.path.join(RUN, f"fold_{f}.pt"))
        if os.path.exists(ck):
            os.remove(ck)
        open(done, "w").close()
        log(f"fold {f + 1}/{nf} finished | mean test Dice {np.mean([float(r[2]) for r in pp]):.4f}")

    if args.import_lopo and os.path.exists(os.path.join(args.import_lopo, "training_history.csv")):
        import shutil
        shutil.copy(os.path.join(args.import_lopo, "training_history.csv"), os.path.join(RUN, "training_history.csv"))
    if args.only_fold < 0 and all(os.path.exists(os.path.join(RUN, f"fold_{f}.done")) for f in run_folds):
        d = {}
        for r in csv.DictReader(open(os.path.join(RUN, "per_patient.csv"))):
            d[r["patient"]] = float(r["dice"])
        v = np.array(list(d.values()))
        log(f"RUN {NAME} COMPLETE | {len(v)} patients | Dice {v.mean():.4f} +/- {v.std(ddof=1):.4f}")
        open(os.path.join(RUN, "ALL_DONE"), "w").close()


if __name__ == "__main__":
    main()
