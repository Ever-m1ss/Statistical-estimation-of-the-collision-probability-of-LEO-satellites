from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.optimize import differential_evolution, minimize
from scipy.special import i0e

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MU_EARTH = 3.986004418e14
R_EARTH = 6371.0e3
R_COLL = 10.0e3


def sanitize(s: str, maxlen: int = 60) -> str:
    """Return a stable ASCII file-name fragment."""
    s = re.sub(r"\s+", "_", str(s).strip())
    s = s.encode("ascii", errors="replace").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:maxlen] if len(s) > maxlen else s


def _tle_float_exponent(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    if "e" in value.lower():
        return float(value)
    m = re.match(r"([ +-]?)(\d+)([+-]\d+)$", value)
    if not m:
        return float(value)
    sign, mant, exp = m.groups()
    return float(f"{sign}0.{mant}") * 10.0 ** int(exp)


def parse_tle_file(path: str) -> list[dict]:
    """Parse a three-line-element file into Keplerian element dictionaries."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = [line.rstrip("\r\n") for line in f if line.strip()]

    sats: list[dict] = []
    i = 0
    while i + 2 < len(raw):
        name, line1, line2 = raw[i].strip(), raw[i + 1], raw[i + 2]
        i += 3
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            continue
        try:
            inc = math.radians(float(line2[8:16]))
            raan = math.radians(float(line2[17:25]))
            ecc = float("0." + line2[26:33].strip())
            argp = math.radians(float(line2[34:42]))
            mean_anomaly = math.radians(float(line2[43:51]))
            mean_motion = float(line2[52:63])
            n_rad_s = mean_motion * 2.0 * pi / 86400.0
            a = (MU_EARTH / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
            sats.append(
                {
                    "name": name,
                    "line1": line1,
                    "line2": line2,
                    "inc": inc,
                    "raan": raan,
                    "ecc": ecc,
                    "argp": argp,
                    "M": mean_anomaly,
                    "mean_motion": mean_motion,
                    "n": n_rad_s,
                    "a": a,
                    "bstar": _tle_float_exponent(line1[53:61]),
                }
            )
        except Exception:
            continue
    return sats


def solve_kepler(M, ecc, tol: float = 1e-12):
    """Solve E - e sin(E) = M for scalar or array M."""
    M_arr = np.asarray(M, dtype=float)
    E = np.mod(M_arr, 2.0 * pi).copy()
    if np.ndim(E) == 0:
        E = float(E)
    for _ in range(50):
        delta = (E - ecc * np.sin(E) - M_arr) / np.maximum(1.0 - ecc * np.cos(E), 1e-14)
        E = E - delta
        if np.max(np.abs(delta)) < tol:
            break
    return E


def kep_to_eci(a, ecc, inc, raan, argp, M):
    """Convert Keplerian elements and mean anomaly to ECI position in meters."""
    M_arr = np.asarray(M, dtype=float)
    E = solve_kepler(M_arr, ecc)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    r = a * (1.0 - ecc * np.cos(E))
    xp = r * np.cos(nu)
    yp = r * np.sin(nu)

    cO, sO = np.cos(raan), np.sin(raan)
    co, so = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc), np.sin(inc)

    x = (cO * co - sO * ci * so) * xp + (-cO * so - sO * ci * co) * yp
    y = (sO * co + cO * ci * so) * xp + (-sO * so + cO * ci * co) * yp
    z = (si * so) * xp + (si * co) * yp
    out = np.stack([x, y, z], axis=-1)
    return out


def _positions(sat: dict, M):
    return kep_to_eci(sat["a"], sat["ecc"], sat["inc"], sat["raan"], sat["argp"], M)


def distance_between(sat1: dict, sat2: dict, M1, M2):
    """Distance between two satellites at the supplied mean anomalies."""
    r1 = _positions(sat1, M1)
    r2 = _positions(sat2, M2)
    d = np.linalg.norm(r1 - r2, axis=-1)
    return float(d) if np.ndim(d) == 0 else d


def find_critical_anomalies(sat1: dict, sat2: dict):
    """Find the anomaly pair minimizing distance over [0, 2*pi)^2."""
    def objective(x):
        return distance_between(sat1, sat2, x[0] % (2.0 * pi), x[1] % (2.0 * pi))

    result_de = differential_evolution(
        objective,
        bounds=[(0.0, 2.0 * pi), (0.0, 2.0 * pi)],
        tol=1e-8,
        polish=False,
        seed=1234,
        updating="immediate",
        workers=1,
    )
    result = minimize(objective, result_de.x, method="Nelder-Mead", options={"maxiter": 500})
    x = result.x if result.success else result_de.x
    m1, m2 = float(x[0] % (2.0 * pi)), float(x[1] % (2.0 * pi))
    return m1, m2, float(objective((m1, m2)))


def mc_collision_estimate(sat1: dict, sat2: dict, N: int, R: float = R_COLL):
    """Uniform Monte Carlo estimate of P(distance < R)."""
    m1 = np.random.uniform(0.0, 2.0 * pi, int(N))
    m2 = np.random.uniform(0.0, 2.0 * pi, int(N))
    hit = distance_between(sat1, sat2, m1, m2) < R
    p_hat = float(np.mean(hit))
    var = float(p_hat * (1.0 - p_hat) / max(int(N), 1))
    return p_hat, var, (m1, m2)


def _wrap_angle(theta):
    return np.mod(theta, 2.0 * pi)


def _angle_diff(theta, mu):
    return (theta - mu + pi) % (2.0 * pi) - pi


def log(x):
    """Numerically safe log."""
    return np.log(np.maximum(x, 1e-300))


def von_mises_log_pdf(theta, mu, kappa):
    """Log PDF of von Mises(theta | mu, kappa)."""
    theta = np.asarray(theta)
    kappa = float(max(kappa, 1e-12))
    return kappa * (np.cos(_angle_diff(theta, mu)) - 1.0) - np.log(2.0 * pi) - np.log(i0e(kappa))


def von_mises_pdf(theta, mu, kappa):
    return np.exp(von_mises_log_pdf(theta, mu, kappa))


def sample_von_mises(mu, kappa, size=1):
    """Sample angles from a von Mises proposal on [0, 2*pi)."""
    return _wrap_angle(np.random.vonmises(mu, max(float(kappa), 1e-9), int(size)))


def importance_sampling_estimate(
    sat1: dict,
    sat2: dict,
    N: int,
    M1_star: float,
    M2_star: float,
    kappa: float,
    R: float = R_COLL,
):
    """Importance-sampling estimate using independent von Mises proposals."""
    N = int(N)
    m1 = sample_von_mises(M1_star, kappa, N)
    m2 = sample_von_mises(M2_star, kappa, N)
    hit = distance_between(sat1, sat2, m1, m2) < R
    log_q = von_mises_log_pdf(m1, M1_star, kappa) + von_mises_log_pdf(m2, M2_star, kappa)
    log_w = -2.0 * np.log(2.0 * pi) - log_q
    contrib = hit.astype(float) * np.exp(np.clip(log_w, -745.0, 700.0))
    p_hat = float(np.mean(contrib))
    var = float(np.var(contrib, ddof=1) / N) if N > 1 else 0.0
    weights = np.exp(np.clip(log_w, -745.0, 700.0))
    ess = float((np.sum(weights) ** 2) / max(np.sum(weights * weights), 1e-300))
    return p_hat, var, (m1, m2, weights, hit), ess


def optimal_kappa(sat1, sat2, M1_star, M2_star, R=R_COLL, n_trial=25):
    """Choose a proposal concentration by a small pilot scan."""
    kappas = np.geomspace(20.0, 5000.0, int(n_trial))
    best = (float(kappas[0]), -np.inf)
    for kappa in kappas:
        p, var, _, ess = importance_sampling_estimate(sat1, sat2, 500, M1_star, M2_star, float(kappa), R)
        score = p / math.sqrt(max(var, 1e-300)) + 1e-3 * ess
        if np.isfinite(score) and score > best[1]:
            best = (float(kappa), float(score))
    return best


def _estimate_proposal_sigma(seed_m1, seed_m2):
    return max(float(np.std(np.r_[seed_m1, seed_m2])) * 0.25, 0.01)


def _mcmc_conditional_batch(
    sat1,
    sat2,
    seeds_m1,
    seeds_m2,
    chain_length,
    threshold,
    proposal_sigma=0.05,
):
    samples_m1, samples_m2 = [], []
    accepts = 0
    total = 0
    for m1, m2 in zip(seeds_m1, seeds_m2):
        cur1, cur2 = float(m1), float(m2)
        for _ in range(int(chain_length)):
            prop1 = float(_wrap_angle(cur1 + np.random.normal(0.0, proposal_sigma)))
            prop2 = float(_wrap_angle(cur2 + np.random.normal(0.0, proposal_sigma)))
            total += 1
            if distance_between(sat1, sat2, prop1, prop2) <= threshold:
                cur1, cur2 = prop1, prop2
                accepts += 1
            samples_m1.append(cur1)
            samples_m2.append(cur2)
    return np.array(samples_m1), np.array(samples_m2), accepts / max(total, 1)


def _autocorrelation_function(x, max_lag=50):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or np.var(x) == 0:
        return np.ones(1)
    x = x - np.mean(x)
    out = []
    denom = np.dot(x, x)
    for lag in range(min(int(max_lag), len(x) - 1) + 1):
        out.append(float(np.dot(x[: len(x) - lag], x[lag:]) / denom))
    return np.array(out)


def _iat(x):
    ac = _autocorrelation_function(x, 50)
    positive = ac[1:][ac[1:] > 0]
    return float(max(1.0, 1.0 + 2.0 * np.sum(positive)))


def subset_simulation_estimate(
    sat1,
    sat2,
    N=3000,
    p0=0.1,
    R=R_COLL,
    max_levels=12,
    verbose=False,
):
    """Subset simulation estimate for rare distance events."""
    N = int(N)
    m1 = np.random.uniform(0.0, 2.0 * pi, N)
    m2 = np.random.uniform(0.0, 2.0 * pi, N)
    cumulative = 1.0
    levels = []
    threshold = np.inf
    for level in range(int(max_levels)):
        d = distance_between(sat1, sat2, m1, m2)
        q = float(np.quantile(d, p0))
        threshold = max(R, q)
        conditional = float(np.mean(d <= threshold))
        cumulative *= conditional
        levels.append(
            {
                "level": level,
                "threshold_km": threshold / 1000.0,
                "conditional_probability": conditional,
                "cumulative_probability": cumulative,
                "acceptance_rate": np.nan if level == 0 else None,
                "iat": _iat(d),
            }
        )
        if verbose:
            print(f"    level {level}: threshold={threshold/1000:.3f} km, p={cumulative:.3e}")
        if threshold <= R:
            break
        idx = np.argsort(d)[: max(2, int(math.ceil(p0 * N)))]
        seeds1, seeds2 = m1[idx], m2[idx]
        chain_length = max(1, math.ceil(N / len(idx)))
        sigma = max(0.01, min(0.5, _estimate_proposal_sigma(seeds1, seeds2)))
        m1, m2, acc = _mcmc_conditional_batch(sat1, sat2, seeds1, seeds2, chain_length, threshold, sigma)
        m1, m2 = m1[:N], m2[:N]
        levels[-1]["acceptance_rate"] = acc

    final_d = distance_between(sat1, sat2, m1, m2)
    if threshold > R:
        final_cond = float(np.mean(final_d < R))
        p_hat = cumulative * final_cond
    else:
        p_hat = cumulative
    cv = math.sqrt(max((1.0 - p0) / max(N * p0, 1), 0.0) * max(len(levels), 1))
    return {
        "p_hat": float(p_hat),
        "cv": float(cv),
        "levels": levels,
        "samples": (m1, m2),
        "distances": final_d,
    }


def analyze_pair(sat1, sat2, N_mc=100000, N_is=5000, N_ss=3000, R=R_COLL, verbose=True):
    """Run all estimators for one satellite pair."""
    name = f"{sat1['name']} vs {sat2['name']}"
    if verbose:
        print(f"\nAnalyzing {name}")
    m1_star, m2_star, d_min = find_critical_anomalies(sat1, sat2)

    t0 = time.time()
    p_mc, var_mc, mc_samples = mc_collision_estimate(sat1, sat2, N_mc, R)
    t_mc = time.time() - t0

    kappa, _ = optimal_kappa(sat1, sat2, m1_star, m2_star, R)
    t0 = time.time()
    p_is, var_is, is_samples, ess = importance_sampling_estimate(sat1, sat2, N_is, m1_star, m2_star, kappa, R)
    t_is = time.time() - t0

    t0 = time.time()
    ss = subset_simulation_estimate(sat1, sat2, N_ss, R=R, verbose=False)
    t_ss = time.time() - t0

    variance_reduction = var_mc / var_is if var_is > 0 and var_mc > 0 else float("nan")
    result = {
        "pair_name": name,
        "sat1": sat1["name"],
        "sat2": sat2["name"],
        "M1_star": m1_star,
        "M2_star": m2_star,
        "d_min": d_min,
        "d_min_km": d_min / 1000.0,
        "P_MC": p_mc,
        "P_IS": p_is,
        "P_SubSim": ss["p_hat"],
        "CV_SubSim": ss["cv"],
        "SubSim_levels": len(ss["levels"]),
        "Var_MC": var_mc,
        "Var_IS": var_is,
        "variance_reduction_IS_over_MC": variance_reduction,
        "ESS": ess,
        "ESS_ratio": ess / max(N_is, 1),
        "kappa_opt": kappa,
        "time_MC_s": t_mc,
        "time_IS_s": t_is,
        "time_SubSim_s": t_ss,
        "mc_samples": mc_samples,
        "is_samples": is_samples,
        "subsim": ss,
    }
    if verbose:
        print(f"  d_min={result['d_min_km']:.3f} km, MC={p_mc:.3e}, IS={p_is:.3e}, SubSim={ss['p_hat']:.3e}")
    return result


def _savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_pair_analysis(result, save_dir):
    pair_dir = os.path.join(save_dir, "pair_analysis")
    os.makedirs(pair_dir, exist_ok=True)
    stem = sanitize(result["pair_name"])

    labels = ["MC", "IS", "SubSim"]
    vals = [result["P_MC"], result["P_IS"], result["P_SubSim"]]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.yscale("log")
    plt.ylabel("Collision probability")
    plt.title(result["pair_name"])
    _savefig(os.path.join(pair_dir, f"pair_{stem}_prob_compare.png"))

    m1, m2 = result["mc_samples"]
    plt.figure(figsize=(5, 5))
    plt.scatter(m1[:5000], m2[:5000], s=3, alpha=0.25)
    plt.xlabel("M1")
    plt.ylabel("M2")
    plt.title("Uniform MC samples")
    _savefig(os.path.join(pair_dir, f"pair_{stem}_mc_scatter.png"))

    is_m1, is_m2, weights, hit = result["is_samples"]
    plt.figure(figsize=(5, 5))
    plt.scatter(is_m1, is_m2, c=np.log10(weights + 1e-300), s=5, cmap="viridis")
    plt.colorbar(label="log10 weight")
    plt.xlabel("M1")
    plt.ylabel("M2")
    plt.title("Importance samples")
    _savefig(os.path.join(pair_dir, f"pair_{stem}_is_scatter.png"))

    plt.figure(figsize=(6, 4))
    plt.hist(np.log10(weights + 1e-300), bins=40, color="#777777")
    plt.xlabel("log10 weight")
    plt.ylabel("count")
    _savefig(os.path.join(pair_dir, f"pair_{stem}_weight_dist.png"))


def plot_subset_simulation(result, save_dir):
    ss_dir = os.path.join(save_dir, "subset_simulation")
    os.makedirs(ss_dir, exist_ok=True)
    stem = sanitize(result["pair_name"])
    levels = result["subsim"]["levels"]
    if not levels:
        return
    x = [lv["level"] for lv in levels]
    thr = [lv["threshold_km"] for lv in levels]
    cum = [lv["cumulative_probability"] for lv in levels]
    cond = [lv["conditional_probability"] for lv in levels]

    plt.figure(figsize=(6, 4))
    plt.plot(x, thr, marker="o")
    plt.axhline(R_COLL / 1000.0, color="red", linestyle="--", label="collision radius")
    plt.xlabel("level")
    plt.ylabel("threshold (km)")
    plt.legend()
    _savefig(os.path.join(ss_dir, f"subsim_{stem}_threshold.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(x, cum, marker="o")
    plt.yscale("log")
    plt.xlabel("level")
    plt.ylabel("cumulative probability")
    _savefig(os.path.join(ss_dir, f"subsim_{stem}_cumulative.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(x, cond, marker="o")
    plt.xlabel("level")
    plt.ylabel("conditional probability")
    _savefig(os.path.join(ss_dir, f"subsim_{stem}_conditional.png"))


def select_diverse_pairs(sats, n_pairs=10, R=R_COLL):
    """Select candidate pairs with small approximate orbital-radius separation."""
    candidates = []
    for i in range(len(sats)):
        for j in range(i + 1, len(sats)):
            da = abs(sats[i]["a"] - sats[j]["a"])
            di = abs(sats[i]["inc"] - sats[j]["inc"])
            score = da + R_EARTH * di
            if score < 250e3:
                candidates.append((score, sats[i], sats[j]))
    candidates.sort(key=lambda x: x[0])
    return [(a, b) for _, a, b in candidates[: int(n_pairs)]]


def _json_safe_result(result):
    return {
        "satellite_pair": result["pair_name"],
        "d_min_km": result["d_min_km"],
        "P_MC": result["P_MC"],
        "P_IS": result["P_IS"],
        "P_SubSim": result["P_SubSim"],
        "CV_SubSim": result["CV_SubSim"],
        "SubSim_levels": result["SubSim_levels"],
        "Var_MC": result["Var_MC"],
        "Var_IS": result["Var_IS"],
        "variance_reduction_IS_over_MC": result["variance_reduction_IS_over_MC"],
        "ESS": result["ESS"],
        "ESS_ratio": result["ESS_ratio"],
        "kappa_opt": result["kappa_opt"],
        "time_MC_s": result["time_MC_s"],
        "time_IS_s": result["time_IS_s"],
        "time_SubSim_s": result["time_SubSim_s"],
    }


def run_full_analysis(tle_path, save_dir, n_pairs=8):
    """Run the full collision-analysis workflow and write JSON/PNG outputs."""
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("LEO collision probability analysis")
    print("=" * 70)
    sats = parse_tle_file(tle_path)
    sats_leo = [s for s in sats if s["a"] < (R_EARTH + 2000e3)]
    print(f"Loaded {len(sats)} satellites; {len(sats_leo)} LEO candidates")
    pairs = select_diverse_pairs(sats_leo, n_pairs)
    print(f"Selected {len(pairs)} pairs")

    results = []
    subsim_rows = []
    for sat1, sat2 in pairs:
        res = analyze_pair(sat1, sat2, N_mc=100000, N_is=3000, N_ss=1500, verbose=True)
        results.append(res)
        plot_pair_analysis(res, save_dir)
        plot_subset_simulation(res, save_dir)
        for lv in res["subsim"]["levels"]:
            row = {"satellite_pair": res["pair_name"], **lv}
            subsim_rows.append(row)

    summary = {
        "total_pairs": len(results),
        "MC_total_samples": len(results) * 100000,
        "IS_total_samples": len(results) * 3000,
        "SubSim_total_samples": len(results) * 1500,
        "results": [_json_safe_result(r) for r in results],
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=True)
    with open(os.path.join(save_dir, "subsim_levels.json"), "w", encoding="utf-8") as f:
        json.dump(subsim_rows, f, ensure_ascii=False, indent=2, allow_nan=True)
    return summary


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    run_full_analysis(os.path.join(base, "3le.txt"), os.path.join(base, "results"), n_pairs=8)
