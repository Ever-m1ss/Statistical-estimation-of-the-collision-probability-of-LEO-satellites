from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm

from collision_analysis import R_COLL, distance_between, find_critical_anomalies, parse_tle_file

TLE_PATH = "3le.txt"
SAVE_DIR = "results"


def _batch_distance(sat1, sat2, M1_arr, M2_arr):
    return distance_between(sat1, sat2, M1_arr, M2_arr)


def calculate_p_true(sat1, sat2, m1_star, m2_star, delta_m=0.08):
    """Local numerical integration around the closest approach."""
    density = 1.0 / (2.0 * np.pi) ** 2

    def integrand(m2, m1):
        return density if distance_between(sat1, sat2, m1, m2) < R_COLL else 0.0

    val, _ = dblquad(
        integrand,
        m1_star - delta_m,
        m1_star + delta_m,
        lambda _m1: m2_star - delta_m,
        lambda _m1: m2_star + delta_m,
        epsabs=1e-12,
    )
    return float(val)


def run_standard_mc(sat1, sat2, N_total=10**6, chunk_size=2 * 10**5):
    hits = 0
    done = 0
    while done < N_total:
        n = min(chunk_size, N_total - done)
        m1 = np.random.uniform(0, 2 * np.pi, n)
        m2 = np.random.uniform(0, 2 * np.pi, n)
        hits += int(np.sum(_batch_distance(sat1, sat2, m1, m2) < R_COLL))
        done += n
    p = hits / max(N_total, 1)
    var = p * (1 - p) / max(N_total, 1)
    se = np.sqrt(var)
    return float(p), float(var), (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))


def run_bias_variance_decomposition(sat1, sat2, p_true, save_dir):
    N_list = [1000, 5000, 20000, 100000]
    K = 20
    rows = []
    for N in N_list:
        vals = []
        for _ in range(K):
            m1 = np.random.uniform(0, 2 * np.pi, N)
            m2 = np.random.uniform(0, 2 * np.pi, N)
            vals.append(float(np.mean(_batch_distance(sat1, sat2, m1, m2) < R_COLL)))
        vals = np.asarray(vals)
        bias2 = float((np.mean(vals) - p_true) ** 2)
        var = float(np.var(vals, ddof=1))
        rows.append({"N": N, "bias2": bias2, "variance": var, "mse": bias2 + var})

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.loglog([r["N"] for r in rows], [r["bias2"] for r in rows], marker="o", label="bias^2")
    plt.loglog([r["N"] for r in rows], [r["variance"] for r in rows], marker="o", label="variance")
    plt.loglog([r["N"] for r in rows], [r["mse"] for r in rows], marker="o", label="MSE")
    plt.xlabel("N")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_variance_converge.png"), dpi=180)
    plt.close()
    return rows


def run_bootstrap_analysis(sat1, sat2, p_true, N=50000, B=1000):
    m1 = np.random.uniform(0, 2 * np.pi, N)
    m2 = np.random.uniform(0, 2 * np.pi, N)
    hits = (_batch_distance(sat1, sat2, m1, m2) < R_COLL).astype(float)
    p_hat = float(np.mean(hits))
    boot = np.mean(np.random.choice(hits, size=(B, N), replace=True), axis=1)
    ci = np.quantile(boot, [0.025, 0.975])
    se = float(np.std(boot, ddof=1))
    z0 = norm.ppf(np.mean(boot < p_hat)) if np.any(boot < p_hat) else 0.0
    return {
        "p_hat": p_hat,
        "p_true": float(p_true),
        "bootstrap_se": se,
        "ci_percentile": [float(ci[0]), float(ci[1])],
        "bias": float(p_hat - p_true),
        "z0": float(z0),
    }


def _find_sat(sats, name):
    for sat in sats:
        if name.upper() in sat["name"].upper():
            return sat
    return None


def find_satellite_pair(sats, p1, p2, fallback=None):
    s1, s2 = _find_sat(sats, p1), _find_sat(sats, p2)
    if s1 is not None and s2 is not None:
        return s1, s2
    if fallback:
        return fallback
    raise ValueError(f"Could not find pair: {p1} vs {p2}")


def main(tle_path=TLE_PATH, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    sats = parse_tle_file(tle_path)
    sat1, sat2 = find_satellite_pair(sats, "0 COSMOS 1721", "0 COSMOS 2129")
    print(f"pair: {sat1['name']} vs {sat2['name']}")
    t0 = time.time()
    m1, m2, dmin = find_critical_anomalies(sat1, sat2)
    p_true = calculate_p_true(sat1, sat2, m1, m2)
    p_mc, var_mc, ci_mc = run_standard_mc(sat1, sat2, N_total=200000)
    bv = run_bias_variance_decomposition(sat1, sat2, p_true, save_dir)
    boot = run_bootstrap_analysis(sat1, sat2, p_true, N=20000, B=300)
    out = {
        "pair": f"{sat1['name']} vs {sat2['name']}",
        "d_min_km": dmin / 1000.0,
        "M1_star": m1,
        "M2_star": m2,
        "p_true_dblquad": p_true,
        "mc": {"p_hat": p_mc, "var": var_mc, "ci95": ci_mc},
        "bias_variance": bv,
        "bootstrap": boot,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(save_dir, "ground_truth_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"d_min={out['d_min_km']:.3f} km, p_true={p_true:.3e}, mc={p_mc:.3e}")
    return out


if __name__ == "__main__":
    main()
