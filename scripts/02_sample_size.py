
from pathlib import Path
import json
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from numpy.polynomial.hermite import hermgauss

ALPHA = 0.05
POWER = 0.80

# Observed NHANES sample statistics.
AVAILABLE_OUTCOME_N = 5210
AVAILABLE_COMPLETE_N = 4448
AVAILABLE_COMPLETE_POS = 549

# Planning assumptions: replace with justified values.
EXPECTED_PREVALENCE = 0.1269
TRAIN_FRACTION = 0.80

RESULTS = {}


# --------------------------------------------------
# RQ1: Correlation between sleep duration and PHQ-9
# --------------------------------------------------

def rq1_correlation(r=0.20):
    z_alpha = norm.ppf(1 - ALPHA / 2)
    z_power = norm.ppf(POWER)

    n = 3 + (
        (z_alpha + z_power) / np.arctanh(abs(r))
    ) ** 2

    return int(np.ceil(n))


RESULTS["RQ1"] = {
    "method": "Fisher z correlation approximation",
    "effect_size_r": 0.20,
    "required_n": rq1_correlation(0.20)
}


# --------------------------------------------------
# RQ2: Logistic regression for sedentary hours
# --------------------------------------------------

def rq2_logistic(
    odds_ratio=1.15,
    prevalence=EXPECTED_PREVALENCE,
    exposure_sd=2.5
):
    """
    Approximate sample size for testing a single
    continuous predictor in logistic regression.

    Assumes normally distributed sedentary hours,
    log-linear odds, and independent observations.
    """

    beta = np.log(odds_ratio)

    # Normal-distribution quadrature nodes.
    nodes, weights = hermgauss(60)
    x = np.sqrt(2) * exposure_sd * nodes
    weights = weights / np.sqrt(np.pi)

    def logistic(z):
        from scipy.special import expit
        return expit(z)

    # Select intercept to match expected prevalence.
    def prevalence_error(intercept):
        p = logistic(intercept + beta * x)
        return np.sum(weights * p) - prevalence

    intercept = brentq(
        prevalence_error, -30, 30
    )

    p = logistic(intercept + beta * x)
    v = p * (1 - p)

    # Per-observation expected information matrix.
    i00 = np.sum(weights * v)
    i01 = np.sum(weights * v * x)
    i11 = np.sum(weights * v * x * x)

    information = np.array([
        [i00, i01],
        [i01, i11]
    ])

    variance_beta = np.linalg.inv(
        information
    )[1, 1]

    z_alpha = norm.ppf(1 - ALPHA / 2)

    def achieved_power(n):
        noncentrality = (
            abs(beta) *
            np.sqrt(n / variance_beta)
        )

        return (
            norm.sf(z_alpha - noncentrality)
            + norm.cdf(-z_alpha - noncentrality)
        )

    required_n = 1
    upper = 100

    while achieved_power(upper) < POWER:
        upper *= 2
        if upper > 1_000_000:
            raise ValueError(
                "Required N exceeds planning limit"
            )

    low = 1
    high = upper

    while low < high:
        mid = (low + high) // 2

        if achieved_power(mid) >= POWER:
            high = mid
        else:
            low = mid + 1

    required_n = low

    return {
        "required_n": required_n,
        "odds_ratio": odds_ratio,
        "exposure_sd_hours": exposure_sd,
        "expected_prevalence": prevalence,
        "achieved_power_approx": achieved_power(
            required_n
        )
    }


RESULTS["RQ2"] = {
    "method": "Logistic Wald information approximation",
    **rq2_logistic()
}


# --------------------------------------------------
# RQ4: ROC-AUC greater than 0.50
# --------------------------------------------------

def auc_variance(auc, n_positive, n_negative):
    """
    Hanley-McNeil large-sample AUC variance.
    """

    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)

    return (
        auc * (1 - auc)
        + (n_positive - 1) * (q1 - auc * auc)
        + (n_negative - 1) * (q2 - auc * auc)
    ) / (n_positive * n_negative)


def rq4_auc(
    expected_auc=0.70,
    null_auc=0.50,
    prevalence=EXPECTED_PREVALENCE
):
    """
    Approximate power for an independently evaluated,
    fixed model, assuming AUC = expected_auc.
    """

    z_alpha = norm.ppf(1 - ALPHA / 2)

    def achieved_power(n):
        n_positive = max(
            2, int(round(n * prevalence))
        )
        n_negative = n - n_positive

        if n_negative < 2:
            return 0.0

        se_null = np.sqrt(
            auc_variance(
                null_auc, n_positive, n_negative
            )
        )

        se_alt = np.sqrt(
            auc_variance(
                expected_auc, n_positive, n_negative
            )
        )

        difference = expected_auc - null_auc
        threshold = z_alpha * se_null

        return (
            norm.sf(
                (threshold - difference) / se_alt
            )
            + norm.cdf(
                (-threshold - difference) / se_alt
            )
        )

    for n in range(20, 100001):
        if achieved_power(n) >= POWER:
            return {
                "required_test_n": n,
                "expected_auc": expected_auc,
                "null_auc": null_auc,
                "expected_positive_cases": round(
                    n * prevalence
                ),
                "achieved_power_approx": achieved_power(n)
            }

    raise ValueError("Required test N not found")


RESULTS["RQ4"] = {
    "method": "Hanley-McNeil AUC power approximation",
    **rq4_auc()
}


# --------------------------------------------------
# Save and print
# --------------------------------------------------

print("\n=== SAMPLE SIZE ANALYSIS ===")

for rq, result in RESULTS.items():
    print(f"\n{rq}")
    for key, value in result.items():
        print(f"  {key}: {value}")

test_n = int(
    np.floor(
        AVAILABLE_COMPLETE_N *
        (1 - TRAIN_FRACTION)
    )
)

test_positive = int(
    round(
        AVAILABLE_COMPLETE_POS *
        (1 - TRAIN_FRACTION)
    )
)

print("\n=== AVAILABLE SAMPLE ===")
print("Outcome-valid N:", AVAILABLE_OUTCOME_N)
print("Complete-case N:", AVAILABLE_COMPLETE_N)
print("Complete-case positives:", AVAILABLE_COMPLETE_POS)
print("Planned test N:", test_n)
print("Approximate test positives:", test_positive)

print("\n=== FEASIBILITY (UNADJUSTED) ===")
print(
    "RQ1:",
    AVAILABLE_OUTCOME_N >= RESULTS["RQ1"]["required_n"]
)
print(
    "RQ2:",
    AVAILABLE_OUTCOME_N >= RESULTS["RQ2"]["required_n"]
)
print(
    "RQ4 test:",
    test_n >= RESULTS["RQ4"]["required_test_n"]
)

output = {
    "alpha": ALPHA,
    "target_power": POWER,
    "assumptions": {
        "prevalence": EXPECTED_PREVALENCE,
        "train_fraction": TRAIN_FRACTION
    },
    "results": RESULTS,
    "available": {
        "outcome_n": AVAILABLE_OUTCOME_N,
        "complete_case_n": AVAILABLE_COMPLETE_N,
        "complete_case_positive": AVAILABLE_COMPLETE_POS,
        "planned_test_n": test_n,
        "approximate_test_positive": test_positive
    }
}

Path("reports").mkdir(exist_ok=True)

with open("reports/sample_size.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved reports/sample_size.json")
