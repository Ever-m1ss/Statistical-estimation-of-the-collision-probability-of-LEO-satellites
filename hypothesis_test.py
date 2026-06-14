from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, chi2, norm


class TestKind(Enum):
    MC_EXACT_BINOMIAL = "MC Exact Binomial"
    MC_SCORE = "MC Score (Wald under H0)"
    MC_LOG_WALD = "MC Log-scale Wald"
    IS_WALD = "IS Wald"
    IS_LOG_WALD = "IS Log-scale Wald"
    FISHER_COMPOSITE = "Fisher Composite (MC+IS)"
    BOOTSTRAP = "Bootstrap Calibrated"


@dataclass
class HypothesisTestConfig:
    p0: float
    alpha: float = 0.05
    test_kind: TestKind = TestKind.MC_EXACT_BINOMIAL
    N_bootstrap: int = 10000
    alternative: str = "greater"


@dataclass
class TestResult:
    test_kind: TestKind
    statistic: float
    p_value: float
    reject_H0: bool
    alpha: float
    p0: float
    p_hat: float
    se_hat: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    N_eff: float = 0.0
    message: str = ""

    @property
    def summary(self) -> str:
        decision = "REJECT" if self.reject_H0 else "not reject"
        return f"[{self.test_kind.value}] stat={self.statistic:.4g}, p={self.p_value:.4g}, {decision}"

    def short_dict(self) -> dict:
        return {
            "test": self.test_kind.value,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "reject_H0": bool(self.reject_H0),
            "alpha": float(self.alpha),
            "p0": float(self.p0),
            "p_hat": float(self.p_hat),
            "se_hat": float(self.se_hat),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "N_eff": float(self.N_eff),
            "message": self.message,
        }


def _normal_ci(p_hat, se, alpha):
    z = norm.ppf(1.0 - alpha / 2.0)
    return max(0.0, p_hat - z * se), min(1.0, p_hat + z * se)


def mc_exact_binomial_test(hits: int, N: int, p0: float, alpha: float = 0.05) -> TestResult:
    hits, N = int(hits), int(N)
    p_hat = hits / N if N > 0 else 0.0
    p_value = float(binom.sf(hits - 1, N, p0)) if N > 0 else 1.0
    se = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / max(N, 1)))
    ci = _normal_ci(p_hat, se, alpha)
    return TestResult(TestKind.MC_EXACT_BINOMIAL, hits, p_value, p_value < alpha, alpha, p0, p_hat, se, *ci, N_eff=N)


def mc_score_test(hits: int, N: int, p0: float, alpha: float = 0.05) -> TestResult:
    p_hat = hits / N if N > 0 else 0.0
    se0 = np.sqrt(max(p0 * (1.0 - p0), 1e-300) / max(N, 1))
    z = (p_hat - p0) / se0
    p_value = float(norm.sf(z))
    se = np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / max(N, 1))
    ci = _normal_ci(p_hat, se, alpha)
    return TestResult(TestKind.MC_SCORE, float(z), p_value, p_value < alpha, alpha, p0, p_hat, float(se), *ci, N_eff=N)


def mc_log_wald_test(hits: int, N: int, p0: float, alpha: float = 0.05, correction: float = 0.5) -> TestResult:
    p_hat = (hits + correction) / (N + 2.0 * correction) if N > 0 else 0.0
    se_log = np.sqrt(1.0 / max(hits + correction, 1e-12) + 1.0 / max(N, 1))
    z = (np.log(max(p_hat, 1e-300)) - np.log(max(p0, 1e-300))) / se_log
    p_value = float(norm.sf(z))
    se = p_hat * se_log
    ci = _normal_ci(p_hat, se, alpha)
    return TestResult(TestKind.MC_LOG_WALD, float(z), p_value, p_value < alpha, alpha, p0, p_hat, float(se), *ci, N_eff=N)


def is_wald_test(p_hat: float, var_hat: float, p0: float, alpha: float = 0.05, N_eff: float = 0.0) -> TestResult:
    se = float(np.sqrt(max(var_hat, 1e-300)))
    z = (p_hat - p0) / se
    p_value = float(norm.sf(z))
    ci = _normal_ci(p_hat, se, alpha)
    return TestResult(TestKind.IS_WALD, float(z), p_value, p_value < alpha, alpha, p0, p_hat, se, *ci, N_eff=N_eff)


def is_log_wald_test(p_hat: float, var_hat: float, p0: float, alpha: float = 0.05, N_eff: float = 0.0) -> TestResult:
    p = max(float(p_hat), 1e-300)
    se_log = np.sqrt(max(var_hat, 1e-300)) / p
    z = (np.log(p) - np.log(max(p0, 1e-300))) / se_log
    p_value = float(norm.sf(z))
    ci = _normal_ci(p_hat, np.sqrt(max(var_hat, 0.0)), alpha)
    return TestResult(TestKind.IS_LOG_WALD, float(z), p_value, p_value < alpha, alpha, p0, p_hat, float(np.sqrt(max(var_hat, 0.0))), *ci, N_eff=N_eff)


def fisher_composite_test(mc_result: TestResult, is_result: TestResult, alpha: float = 0.05) -> TestResult:
    pvals = np.clip([mc_result.p_value, is_result.p_value], 1e-300, 1.0)
    stat = float(-2.0 * np.sum(np.log(pvals)))
    p_value = float(chi2.sf(stat, 2 * len(pvals)))
    return TestResult(
        TestKind.FISHER_COMPOSITE,
        stat,
        p_value,
        p_value < alpha,
        alpha,
        mc_result.p0,
        max(mc_result.p_hat, is_result.p_hat),
        message="Fisher combination of MC and IS p-values",
    )


def bootstrap_test(samples, p0: float, alpha: float = 0.05, N_bootstrap: int = 10000) -> TestResult:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return TestResult(TestKind.BOOTSTRAP, 0.0, 1.0, False, alpha, p0, 0.0, message="empty sample")
    p_hat = float(np.mean(arr))
    boot = np.mean(np.random.choice(arr, size=(int(N_bootstrap), arr.size), replace=True), axis=1)
    p_value = float(np.mean(boot <= p0)) if p_hat > p0 else 1.0
    ci = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    se = float(np.std(boot, ddof=1))
    z = (p_hat - p0) / max(se, 1e-300)
    return TestResult(TestKind.BOOTSTRAP, float(z), p_value, p_value < alpha, alpha, p0, p_hat, se, float(ci[0]), float(ci[1]), N_eff=arr.size)


def run_mc_tests(hits: int, N: int, p0: float, alpha: float = 0.05):
    return [
        mc_exact_binomial_test(hits, N, p0, alpha),
        mc_score_test(hits, N, p0, alpha),
        mc_log_wald_test(hits, N, p0, alpha),
    ]


def run_is_tests(p_hat: float, var_hat: float, p0: float, alpha: float = 0.05, N_eff: float = 0.0):
    return [
        is_wald_test(p_hat, var_hat, p0, alpha, N_eff),
        is_log_wald_test(p_hat, var_hat, p0, alpha, N_eff),
    ]


def run_all_tests(
    hits: int,
    N_mc: int,
    p_is: float,
    var_is: float,
    p0: float,
    alpha: float = 0.05,
    N_eff: float = 0.0,
):
    mc = run_mc_tests(hits, N_mc, p0, alpha)
    iss = run_is_tests(p_is, var_is, p0, alpha, N_eff)
    return mc + iss + [fisher_composite_test(mc[0], iss[0], alpha)]


@dataclass
class PowerResult:
    N: int
    p_true: float
    p0: float
    alpha: float
    power: float
    test_kind: str


def power_curve_mc_exact_binomial(p_true, p0, N_values, alpha=0.05):
    out = []
    for N in N_values:
        crit = int(binom.isf(alpha, int(N), p0)) + 1
        power = float(binom.sf(crit - 1, int(N), p_true))
        out.append(PowerResult(int(N), p_true, p0, alpha, power, TestKind.MC_EXACT_BINOMIAL.value))
    return out


def power_curve_mc_score(p_true, p0, N_values, alpha=0.05):
    z_alpha = norm.ppf(1.0 - alpha)
    out = []
    for N in N_values:
        se0 = np.sqrt(max(p0 * (1.0 - p0), 1e-300) / int(N))
        threshold = p0 + z_alpha * se0
        se1 = np.sqrt(max(p_true * (1.0 - p_true), 1e-300) / int(N))
        power = float(norm.sf((threshold - p_true) / se1))
        out.append(PowerResult(int(N), p_true, p0, alpha, power, TestKind.MC_SCORE.value))
    return out


def power_curve_is_wald(p_true, p0, N_values, alpha=0.05, cv=0.2):
    z_alpha = norm.ppf(1.0 - alpha)
    out = []
    for N in N_values:
        se = max(abs(p_true) * cv * np.sqrt(1000.0 / int(N)), 1e-300)
        power = float(norm.sf(z_alpha - (p_true - p0) / se))
        out.append(PowerResult(int(N), p_true, p0, alpha, power, TestKind.IS_WALD.value))
    return out


def minimum_sample_size(p_true, p0, alpha=0.05, target_power=0.8, method="mc_score", max_N=10_000_000):
    if p_true <= p0:
        return None
    grid = np.unique(np.logspace(2, np.log10(max_N), 120).astype(int))
    if method.lower().startswith("is"):
        curve = power_curve_is_wald(p_true, p0, grid, alpha)
    elif "exact" in method.lower():
        curve = power_curve_mc_exact_binomial(p_true, p0, grid, alpha)
    else:
        curve = power_curve_mc_score(p_true, p0, grid, alpha)
    for item in curve:
        if item.power >= target_power:
            return item.N
    return None


def sample_size_vs_effect(p0, effects, alpha=0.05, target_power=0.8, method="mc_score"):
    return [{"p_true": float(p0 * e), "effect": float(e), "N": minimum_sample_size(p0 * e, p0, alpha, target_power, method)} for e in effects]


def run_hypothesis_test_on_pair(pair_result: dict, p0: float = 1e-7, alpha: float = 0.05):
    p_mc = float(pair_result.get("P_MC", 0.0))
    n_mc = int(pair_result.get("N_MC", pair_result.get("N_mc", 100000)))
    hits = int(round(p_mc * n_mc))
    p_is = float(pair_result.get("P_IS", 0.0))
    var_is = float(pair_result.get("Var_IS", 0.0))
    ess = float(pair_result.get("ESS", 0.0))
    return run_all_tests(hits, n_mc, p_is, var_is, p0, alpha, ess)


def _plot_curve(items, label):
    plt.plot([x.N for x in items], [x.power for x in items], marker="o", label=label)


def plot_power_curves(p_true, p0, save_path, alpha=0.05):
    Ns = np.unique(np.logspace(2, 6, 30).astype(int))
    plt.figure(figsize=(7, 4))
    _plot_curve(power_curve_mc_score(p_true, p0, Ns, alpha), "MC score")
    _plot_curve(power_curve_is_wald(p_true, p0, Ns, alpha), "IS Wald")
    plt.xscale("log")
    plt.ylim(0, 1.02)
    plt.xlabel("N")
    plt.ylabel("power")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_test_comparison(results, save_path):
    plt.figure(figsize=(8, 4))
    labels = [r.test_kind.value for r in results]
    pvals = [r.p_value for r in results]
    plt.bar(range(len(labels)), pvals)
    plt.axhline(results[0].alpha if results else 0.05, color="red", linestyle="--")
    plt.yscale("log")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("p-value")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_sample_size_curves(rows, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot([r["effect"] for r in rows], [r["N"] or np.nan for r in rows], marker="o")
    plt.yscale("log")
    plt.xlabel("effect multiplier")
    plt.ylabel("minimum N")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def demo_from_summary(summary_path="results/summary.json", save_dir="results/hypothesis_tests"):
    with open(summary_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    rows = data.get("results") or data.get("缁撴灉") or []
    if not rows:
        return []
    row = rows[0]
    results = run_hypothesis_test_on_pair(row)
    os.makedirs(save_dir, exist_ok=True)
    plot_test_comparison(results, os.path.join(save_dir, "test_comparison_pvalues.png"))
    return results


def demo_full_analysis():
    return demo_from_summary()


if __name__ == "__main__":
    for result in demo_full_analysis():
        print(result.summary)
