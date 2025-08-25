import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import matplotlib.transforms as mtransforms

POWER_RE = re.compile(r'power\s*=\s*(\d+)\s*')
SEP_RE = re.compile(r'^\s*={3,}\s*$')

def _to_num(val: str):
    s = val.strip()
    if s.endswith("ms"):
        s = s[:-2].strip()
    if s.endswith("x"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        try:
            return int(s)
        except ValueError:
            return val.strip()

def parse_summary_file(path):
    results = []
    current = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if SEP_RE.match(line):
                continue

            if line.startswith("Summary over power"):
                if current and "power" in current and "Matrix Size" in current and "Diagonal Size" in current:
                    results.append(current)
                current = {}
                m = POWER_RE.search(line)
                if m:
                    current["power"] = int(m.group(1))
                continue

            if current is not None and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                current[key] = _to_num(val)

    if current and "power" in current and "Matrix Size" in current and "Diagonal Size" in current:
        results.append(current)

    return results

def plot_per_diag_power(results, outdir="plots"):
    import os
    os.makedirs(outdir, exist_ok=True)

    groups = defaultdict(list)
    for r in results:
        try:
            diag = int(r["Diagonal Size"])
            pwr  = int(r["power"])
            groups[(diag, pwr)].append(r)
        except (KeyError, ValueError):
            continue

    for (diag, pwr), rows in sorted(groups.items()):
        rows = [r for r in rows if "Yours (kernel+copy)" in r and "cuSPARSE (compute+copy)" in r and "Ratio (cuSPARSE / Ours)" in r]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["Matrix Size"]))

        sizes = [int(r["Matrix Size"]) for r in rows]
        ours  = [float(r["Yours (kernel+copy)"]) for r in rows]
        cus   = [float(r["cuSPARSE (compute+copy)"]) for r in rows]
        ratio = [float(r["Ratio (cuSPARSE / Ours)"]) for r in rows]

        x = range(len(sizes))
        bw = 0.35

        fig, ax1 = plt.subplots(figsize=(9, 6))

        b1 = ax1.bar([i - bw/2 for i in x], ours, width=bw, label="Ours")
        b2 = ax1.bar([i + bw/2 for i in x], cus,  width=bw, label="cuSPARSE")
        ax1.set_xlabel("Matrix Size (N)")
        ax1.set_ylabel("Time (ms)")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels([str(s) for s in sizes])
        ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        l1, = ax2.plot(list(x), ratio, marker="o", linestyle="-", color="red", label="Speedup")
        ax2.set_ylabel("Speedup (x)")

        for xi, yi in zip(x, ratio):
            # create a small upward offset in points (3 pt)
            trans = ax2.transData + mtransforms.ScaledTranslation(0, 3/72., fig.dpi_scale_trans)
            ax2.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom",
                    fontsize=12, color="black", transform=trans)
        handles = [b1, b2, l1]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="best")

        plt.title(f"Diagonal Size = {diag}  |  power = {pwr}")
        plt.tight_layout()

        # save to PDF (high resolution)
        fname = os.path.join(outdir, f"diag{diag}_power{pwr}.pdf")
        plt.savefig(fname, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")



def plot_by_diag_compare_powers(results, powers=(3, 5), outdir="plots"):
    import os
    import matplotlib.pyplot as plt
    from collections import defaultdict

    os.makedirs(outdir, exist_ok=True)

    # index data: {(diag, power, size) -> (ours_total_ms, cus_ms)}
    data = {}
    diags = set()
    sizes_by_diag = defaultdict(set)

    for r in results:
        try:
            diag = int(r["Diagonal Size"])
            pwr  = int(r["power"])
            size = int(r["Matrix Size"])
        except (KeyError, ValueError):
            continue

        if "Yours (kernel+copy)" in r and "cuSPARSE (compute+copy)" in r:
            ours = float(r["Yours (kernel+copy)"])
            cus  = float(r["cuSPARSE (compute+copy)"])
            data[(diag, pwr, size)] = (ours, cus)
            diags.add(diag)
            sizes_by_diag[diag].add(size)

    if not diags:
        print("No valid rows to plot.")
        return

    for diag in sorted(diags):
        # union of sizes across all powers (so bars align)
        sizes = sorted(sizes_by_diag[diag])
        if not sizes:
            continue

        # Prepare series for each metric/power; fill missing with 0
        def get_series(pwr):
            ours = []
            cus  = []
            for s in sizes:
                ours_val, cus_val = data.get((diag, pwr, s), (0.0, 0.0))
                ours.append(ours_val)
                cus.append(cus_val)
            return ours, cus

        ours_p3, cus_p3 = get_series(powers[0])
        ours_p5, cus_p5 = get_series(powers[1])

        x = list(range(len(sizes)))
        # 4 bars per group -> choose a narrow width to avoid overlap
        bw = 0.18
        offsets = (-1.5*bw, -0.5*bw, 0.5*bw, 1.5*bw)

        fig, ax = plt.subplots(figsize=(10, 6))

        b1 = ax.bar([i + offsets[0] for i in x], ours_p3, width=bw, label=f"Ours (p={powers[0]})")
        b2 = ax.bar([i + offsets[1] for i in x], cus_p3,  width=bw, label=f"cuSPARSE (p={powers[0]})")
        b3 = ax.bar([i + offsets[2] for i in x], ours_p5, width=bw, label=f"Ours (p={powers[1]})")
        b4 = ax.bar([i + offsets[3] for i in x], cus_p5,  width=bw, label=f"cuSPARSE (p={powers[1]})")

        ax.set_xlabel("Matrix Size (N)")
        ax.set_ylabel("Time (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        ax.legend(loc="best")
        ax.set_title(f"Diagonal Size = {diag} | Compare power={powers[0]} vs power={powers[1]}")

        plt.tight_layout()
        outpath = os.path.join(outdir, f"diag{diag}_p{powers[0]}p{powers[1]}_bySize.pdf")
        plt.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outpath}")
def plot_all_fixed_size_power(results, outdir="plots"):
    """
    Auto-draw one figure per (Matrix Size, power).
    X-axis = Diagonal Size.
    Bars = Ours (kernel+copy) and cuSPARSE (compute+copy).
    Line = Ratio (cuSPARSE / Ours).
    Missing data are treated as 0.
    """


    os.makedirs(outdir, exist_ok=True)

    # Index results: {(msize, power): {diag: (ours_ms, cus_ms, ratio)}}
    by_mp = defaultdict(dict)
    all_pairs = set()
    diags_by_pair = defaultdict(set)

    for r in results:
        try:
            msize = int(r["Matrix Size"])
            pwr   = int(r["power"])
            diag  = int(r["Diagonal Size"])
        except (KeyError, ValueError):
            continue

        ours  = float(r.get("Yours (kernel+copy)", 0.0))
        cus   = float(r.get("cuSPARSE (compute+copy)", 0.0))
        ratio = float(r.get("Ratio (cuSPARSE / Ours)", 0.0))

        by_mp[(msize, pwr)][diag] = (ours, cus, ratio)
        all_pairs.add((msize, pwr))
        diags_by_pair[(msize, pwr)].add(diag)

    if not all_pairs:
        print("No valid (Matrix Size, power) pairs found.")
        return

    for (msize, pwr) in sorted(all_pairs):
        diag_map = by_mp[(msize, pwr)]
        # Union of diagonals for this (msize, pwr)
        diags = sorted(diags_by_pair[(msize, pwr)])
        if not diags:
            continue

        # Build aligned series; missing -> 0
        ours  = []
        cus   = []
        ratio = []
        for d in diags:
            o, c, r = diag_map.get(d, (0.0, 0.0, 0.0))
            ours.append(o)
            cus.append(c)
            ratio.append(r)

        x = range(len(diags))
        bw = 0.35

        fig, ax1 = plt.subplots(figsize=(9, 6))

        b1 = ax1.bar([i - bw/2 for i in x], ours, width=bw, label="Ours (kernel+copy)")
        b2 = ax1.bar([i + bw/2 for i in x], cus,  width=bw, label="cuSPARSE (compute+copy)")
        ax1.set_xlabel("Diagonal Size")
        ax1.set_ylabel("Time (ms)")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels([str(d) for d in diags])
        ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        l1, = ax2.plot(list(x), ratio, marker="o", linestyle="-", color="red", label="Speedup (cuSPARSE / Ours)")
        ax2.set_ylabel("Speedup (x)")

        # label ratio points slightly above markers (use 5% max for offset; handle all-zero safely)
        for xi, yi in zip(x, ratio):
            # create a small upward offset in points (3 pt)
            trans = ax2.transData + mtransforms.ScaledTranslation(0, 3/72., fig.dpi_scale_trans)
            ax2.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom",
                    fontsize=12, color="black", transform=trans)

        # combine legends
        handles = [b1, b2, l1]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="best")

        plt.title(f"Matrix Size = {msize}  |  power = {pwr}")
        plt.tight_layout()

        # save high-res PDF
        fname = os.path.join(outdir, f"size{msize}_power{pwr}_byDiag.pdf")
        plt.savefig(fname, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == "__main__":
    log_path = "summary.txt"  # change to your file
    data = parse_summary_file(log_path)
    plot_per_diag_power(data)
    plot_by_diag_compare_powers(data)
    plot_all_fixed_size_power(data)
