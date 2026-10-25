#!/usr/bin/env python3
"""Fig_spiking panel (b): log-scale energy axis and separate label offsets so the two SNN points/labels do not overlap.
Usage:  cd ~/dataset && python3 patch_spiking.py"""
import shutil

P = "analyze.py"
s = open(P).read()
shutil.copy(P, P + ".bak_spiking")

old = 'fontsize=4.5, xytext=(3, 3), textcoords="offset points")'
new = 'fontsize=4.5, xytext=((3, -16) if n == args.dwi_only else (3, 4)), textcoords="offset points")'
assert s.count(old) == 1, "annotate line not found exactly once: %d" % s.count(old)
s = s.replace(old, new)

old = 'axs[1].set_xlabel("Estimated energy / slice (uJ)")'
assert s.count(old) == 1, "xlabel line not found exactly once: %d" % s.count(old)
s = s.replace(old, 'axs[1].set_xscale("log"); ' + old)

open(P, "w").write(s)
print("patched. backup:", P + ".bak_spiking")
