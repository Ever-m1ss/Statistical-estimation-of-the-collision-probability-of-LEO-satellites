from __future__ import annotations

import json
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from collision_analysis import (
    R_COLL,
    find_critical_anomalies,
    importance_sampling_estimate,
    mc_collision_estimate,
    optimal_kappa,
    parse_tle_file,
)

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
TLE_PATH = os.path.join(BASE, "3le.txt")
SAVE_DIR = os.path.join(BASE, "results")

PAIR_DEFS = [
    ("0 QIANFAN-25", "0 QIANFAN-149"),
    ("0 GLOBAL-34", "0 STRIX-4"),
    ("0 YAOGAN-39 03C", "0 YAOGAN-39 04B"),
    ("0 LASARSAT", "0 CROCUBE"),
    ("0 COSMOS 1721", "0 COSMOS 2129"),
    ("0 SAOCOM 1-B", "0 BEIJING 3B"),
    ("0 STARLINK-2081", "0 STARLINK-31293"),
]

N_REPLICATIONS_IS = 30
N_REPLICATIONS_MC = 10
SAMPLE_SIZES_IS = [200, 500, 1000, 2000, 5000]
SAMPLE_SIZES_MC = [5000, 10000, 20000]
N_THRESHOLDS = 100


def _find_sat(sats, name):
    needle = name.upper()
    for sat in sats:
        if needle in sat["name"].upper():
            return sat
    return None


def prepare_pairs(tle_path=TLE_PATH, target_names=None):
    sats = parse_tle_file(tle_path)
    pairs = []
    for left, right in PAIR_DEFS:
        s1, s2 = _find_sat(sats, left), _find_sat(sats, right)
        if s1 is not None and s2 is not None:
            pairs.append((s1, s2))
            print(f"pair: {s1['name']} vs {s2['name']}")
    return pairs


def compute_ground_truth(pairs):
    ground_truth = []
    for idx, (s1, s2) in enumerate(pairs):
        m1, m2, dmin = find_critical_anomalies(s1, s2)
        kappa, _ = optimal_kappa(s1, s2, m1, m2)
        p_ref, var_ref, _, ess = importance_sampling_estimate(s1, s2, 20000, m1, m2, kappa)
        if dmin > R_COLL:
            p_ref = 0.0
        entry = {
            "idx": idx,
            "name": f"{s1['name']} vs {s2['name']}",
            "d_min_km": dmin / 1000.0,
            "is_collision": bool(dmin < R_COLL),
            "Pc_ref": float(p_ref),
            "Pc_ref_var": float(var_ref),
            "kappa_opt": float(kappa),
            "M1_star": float(m1),
            "M2_star": float(m2),
            "ESS": float(ess),
            "s1": s1,
            "s2": s2,
        }
        ground_truth.append(entry)
        print(f"[{idx}] {entry['name']}: d_min={entry['d_min_km']:.3f} km, Pc_ref={entry['Pc_ref']:.3e}")
    return ground_truth


def run_simulation(
    ground_truth,
    n_reps_is=N_REPLICATIONS_IS,
    n_reps_mc=N_REPLICATIONS_MC,
    sample_sizes_is=SAMPLE_SIZES_IS,
    sample_sizes_mc=SAMPLE_SIZES_MC,
):
    rows = []
    for gt in ground_truth:
        s1, s2 = gt["s1"], gt["s2"]
        for N in sample_sizes_is:
            for rep in range(int(n_reps_is)):
                p, var, _, ess = importance_sampling_estimate(s1, s2, N, gt["M1_star"], gt["M2_star"], gt["kappa_opt"])
                rows.append({"pair_idx": gt["idx"], "method": "IS", "N": int(N), "rep": rep, "p_hat": p, "var_hat": var, "ESS": ess, "truth": gt["is_collision"]})
        for N in sample_sizes_mc:
            for rep in range(int(n_reps_mc)):
                p, var, _ = mc_collision_estimate(s1, s2, N)
                rows.append({"pair_idx": gt["idx"], "method": "MC", "N": int(N), "rep": rep, "p_hat": p, "var_hat": var, "ESS": 0.0, "truth": gt["is_collision"]})
    return rows


def compute_confusion_matrix(p_hats_H0, p_hats_H1, threshold):
    h0 = np.asarray(p_hats_H0)
    h1 = np.asarray(p_hats_H1)
    fp = int(np.sum(h0 > threshold))
    tn = int(np.sum(h0 <= threshold))
    tp = int(np.sum(h1 > threshold))
    fn = int(np.sum(h1 <= threshold))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "FPR": fp / max(fp + tn, 1), "TPR": tp / max(tp + fn, 1)}


def _auc(fpr, tpr):
    order = np.argsort(fpr)
    return float(np.trapz(np.asarray(tpr)[order], np.asarray(fpr)[order]))


def analyze_results(all_results, n_thresholds=N_THRESHOLDS):
    analysis = []
    methods = sorted(set(r["method"] for r in all_results))
    for method in methods:
        Ns = sorted(set(r["N"] for r in all_results if r["method"] == method))
        for N in Ns:
            rows = [r for r in all_results if r["method"] == method and r["N"] == N]
            h0 = [r["p_hat"] for r in rows if not r["truth"]]
            h1 = [r["p_hat"] for r in rows if r["truth"]]
            if not h0 or not h1:
                continue
            lo, hi = min(h0 + h1), max(h0 + h1)
            thresholds = np.linspace(lo, hi + 1e-300, int(n_thresholds))
            cms = [compute_confusion_matrix(h0, h1, t) for t in thresholds]
            fpr = [c["FPR"] for c in cms]
            tpr = [c["TPR"] for c in cms]
            analysis.append(
                {
                    "method": method,
                    "N": int(N),
                    "n_H0": len(h0),
                    "n_H1": len(h1),
                    "p_hat_H0_mean": float(np.mean(h0)),
                    "p_hat_H0_std": float(np.std(h0)),
                    "p_hat_H1_mean": float(np.mean(h1)),
                    "p_hat_H1_std": float(np.std(h1)),
                    "auc": _auc(fpr, tpr),
                    "power_at_5pct_fpr": max((c["TPR"] for c in cms if c["FPR"] <= 0.05), default=0.0),
                    "power_at_1pct_fpr": max((c["TPR"] for c in cms if c["FPR"] <= 0.01), default=0.0),
                    "roc": [{"threshold": float(t), **c} for t, c in zip(thresholds, cms)],
                }
            )
    return analysis


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_roc_curves(analysis, save_dir, ground_truth=None):
    out = os.path.join(save_dir, "statistical_testing")
    _ensure_dir(out)
    plt.figure(figsize=(6, 5))
    for item in analysis:
        roc = item["roc"]
        plt.plot([r["FPR"] for r in roc], [r["TPR"] for r in roc], label=f"{item['method']} N={item['N']}")
    plt.xlabel("False positive rate")
    plt.ylabel("Detection rate")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "roc_detection_rate.png"), dpi=180)
    plt.close()


def plot_power_curves(analysis, save_dir):
    out = os.path.join(save_dir, "statistical_testing")
    _ensure_dir(out)
    plt.figure(figsize=(6, 4))
    for method in sorted(set(a["method"] for a in analysis)):
        rows = sorted([a for a in analysis if a["method"] == method], key=lambda x: x["N"])
        plt.plot([r["N"] for r in rows], [r["power_at_5pct_fpr"] for r in rows], marker="o", label=method)
    plt.xscale("log")
    plt.ylim(0, 1.02)
    plt.xlabel("N")
    plt.ylabel("Power at 5% FPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "power_vs_n.png"), dpi=180)
    plt.close()


def plot_bias_and_ci(ground_truth, all_results, analysis, save_dir):
    out = os.path.join(save_dir, "statistical_testing")
    _ensure_dir(out)
    truth = {g["idx"]: g["Pc_ref"] for g in ground_truth}
    labels, bias = [], []
    for method in sorted(set(r["method"] for r in all_results)):
        rows = [r for r in all_results if r["method"] == method]
        vals = []
        for idx in sorted(truth):
            pr = [r["p_hat"] for r in rows if r["pair_idx"] == idx]
            if pr:
                vals.append(np.mean(pr) - truth[idx])
        labels.append(method)
        bias.append(float(np.mean(vals)) if vals else 0.0)
    plt.figure(figsize=(5, 4))
    plt.bar(labels, bias)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("mean bias")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "bias_relative_bias.png"), dpi=180)
    plt.close()


def print_summary_table(analysis, ground_truth=None):
    print("\nmethod     N        AUC     power@5%")
    print("--------------------------------------")
    for a in sorted(analysis, key=lambda x: (x["method"], x["N"])):
        print(f"{a['method']:<8} {a['N']:<8} {a['auc']:<7.3f} {a['power_at_5pct_fpr']:<7.3f}")


def main():
    t0 = time.time()
    os.makedirs(SAVE_DIR, exist_ok=True)
    pairs = prepare_pairs(TLE_PATH)
    ground_truth = compute_ground_truth(pairs)
    all_results = run_simulation(ground_truth)
    analysis = analyze_results(all_results)

    public_gt = [{k: v for k, v in g.items() if k not in ("s1", "s2")} for g in ground_truth]
    with open(os.path.join(SAVE_DIR, "statistical_testing_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ground_truth": public_gt, "analysis_summary": [{k: v for k, v in a.items() if k != "roc"} for a in analysis]}, f, ensure_ascii=False, indent=2)

    plot_roc_curves(analysis, SAVE_DIR, ground_truth)
    plot_power_curves(analysis, SAVE_DIR)
    plot_bias_and_ci(ground_truth, all_results, analysis, SAVE_DIR)
    print_summary_table(analysis, ground_truth)
    print(f"done in {time.time() - t0:.1f}s")
    return analysis


if __name__ == "__main__":
    main()
