#!/usr/bin/env python3
"""Fig_spiking fixes (run AFTER patch_spiking.py):
 1. panel (b): widen the y-range so the SNN DWI label no longer runs into the x tick labels
 2. panel (c) "effect of time steps" needs >= 2 SNN DWI+ADC runs with different T; only T=2 exists,
    so the panel is blank. Build a 2-panel figure in that case instead of a 3-panel one with an empty slot.
Usage:  cd ~/dataset && python3 patch_spiking2.py"""
import shutil

P = "analyze.py"
s = open(P).read()
shutil.copy(P, P + ".bak_spiking2")

# 1. y-range of panel (b)
old = 'axs[1].set_xscale("log"); '
assert s.count(old) == 1, "xscale line not found exactly once: %d (run patch_spiking.py first)" % s.count(old)
new = old + ('_m = [arr(n, "dice").mean() for n in disp if n in E]; '
             'axs[1].set_ylim(min(_m) - 0.03, max(_m) + 0.015); ')
s = s.replace(old, new)

# 2. two panels when there is no time-step comparison
old = "fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.5))"
assert s.count(old) == 1, "subplots line not found exactly once: %d" % s.count(old)
new = ('_hasT = sum(1 for n in runs if n.startswith("snn") and "adc" in n and "_k3_" in n) >= 2\n'
       'fig, axs = plt.subplots(1, 3 if _hasT else 2, figsize=(6.9 if _hasT else 4.7, 2.5))')
s = s.replace(old, new)

old = 'axs[2].axis("off")'
assert s.count(old) == 1, "axis-off line not found exactly once: %d" % s.count(old)
s = s.replace(old, "pass")

open(P, "w").write(s)
print("patched. backup:", P + ".bak_spiking2")
