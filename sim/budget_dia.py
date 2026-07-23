import sys, numpy as np
inp, outp, B = sys.argv[1], sys.argv[2], int(sys.argv[3])  # B = #conjugate-pair groups to keep
lines = open(inp).read().splitlines()
hdr = lines[0].split(); N = int(hdr[1])
diags = []  # (offset, vals-str, norm2)
for ln in lines[1:]:
    if ':' not in ln: continue
    o, vs = ln.split(':', 1); o = int(o)
    v = np.fromstring(vs, sep=' '); diags.append((o, vs.strip(), float(np.dot(v, v))))
# group by |offset|, rank groups by summed norm, keep top-B groups
from collections import defaultdict
g = defaultdict(list)
for i, (o, vs, nn) in enumerate(diags): g[abs(o)].append(i)
gnorm = {k: sum(diags[i][2] for i in idx) for k, idx in g.items()}
keepk = set(sorted(gnorm, key=lambda k: -gnorm[k])[:B])
kept = [diags[i] for k in keepk for i in g[k]]
kept.sort(key=lambda t: t[0])
with open(outp, 'w') as f:
    f.write(f"N {N} D {len(kept)}\n")
    for o, vs, _ in kept: f.write(f"{o}: {vs}\n")
print(f"{outp}: kept {B} groups -> {len(kept)} diagonals")
