import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gaussian_kde

# Load our custom matplotlib style file.
current_dir = os.path.dirname(__file__)
plt.style.use(os.path.join(current_dir, "customstyle.mplstyle"))


def GCV(values):
    "Calculate the geometric coefficient of variation"
    s = np.std(np.log(values))
    gcv = np.sqrt(np.exp(s**2) - 1)
    return gcv


def CI_G_LD(
    GM_observed,
    GM_predicted,
    GCV_observed,
    GCV_predicted,
    N_observed,
    N_predicted,
    alpha=0.1,
):
    """
    Calculate CI using the GLM approach and considering variance of predictions.

    This function calculates a CI of the difference in means on logscale and
    then transform this back to the original scale. A 2-sided unpaired
    t-test assuming unequal variances is used.

    Arguments:
        GM_observed: Observed geometric mean
        GM_predicted: Predicted geometric mean
        GCV_observed: Observed geometric coefficient of variation
        GCV_predicted: Predicted geometric coefficient of variation
        N_observed: Number of observed subjects
        N_predicted: Number of predicted subjects
        alpha: Confidence level. The default value of 0.1 corresponds to 90% CI.

    Returns:
        CI: The (1-alpha)*100% confidence interval.
    """
    if type(N_observed) != int:
        raise ValueError("N_observed must be an integer, do not use dots.")
    if type(N_predicted) != int:
        raise ValueError("N_predicted must be an integer, do not use dots.")
    average_observed = np.log(GM_observed)
    average_predicted = np.log(GM_predicted)
    var_observed = np.log(GCV_observed**2 + 1)
    var_predicted = np.log(GCV_predicted**2 + 1)
    var_total = var_observed / N_observed + var_predicted / N_predicted
    df = (var_total) ** 2 / (
        (var_observed / N_observed) ** 2 / (N_observed - 1)
        + (var_predicted / N_predicted) ** 2 / (N_predicted - 1)
    )
    t = scipy.stats.t(df).ppf(1 - alpha / 2)
    loglowerbound = average_predicted - average_observed - t * np.sqrt(var_total)
    logupperbound = average_predicted - average_observed + t * np.sqrt(var_total)
    CI = [np.round(np.exp(loglowerbound), 2), np.round(np.exp(logupperbound), 2)]
    return np.round(GM_predicted / GM_observed, 2), CI


def CRPS_metrics(observations, model_1_predictions, model_2_predictions=None, B=10000):
    """Calculate average CRPS with confidence intervals and skill scores."""
    CRPS_values_1 = []
    skills_CRPS_1 = []
    min_value = min([min(observations), min(model_1_predictions)])
    max_value = max([max(observations), max(model_1_predictions)])
    if model_2_predictions is not None:
        skills_CRPS_2 = []
        CRPS_values_2 = []
        min_value = min([min(observations), min(model_1_predictions), min(model_2_predictions)])
        max_value = max([max(observations), max(model_1_predictions), max(model_2_predictions)])
    x_values = np.linspace(min_value, max_value, 800)
    for _ in range(B):
        observations_resample = np.random.choice(observations, size=len(observations), replace=True)
        model_1_predictions_resample = np.random.choice(
            model_1_predictions, size=len(observations), replace=True
        )

        cdf_1 = scipy.stats.ecdf(model_1_predictions_resample).cdf.evaluate
        cdf_observations = scipy.stats.ecdf(observations_resample).cdf.evaluate
        cdf_naive = scipy.stats.ecdf([np.median(observations_resample)]).cdf.evaluate

        CRPS_naive = np.trapz(
            cdf_naive(x_values) ** 2
            - 2 * cdf_naive(x_values) * cdf_observations(x_values)
            + cdf_observations(x_values),
            x_values,
        )
        CRPS_1 = np.trapz(
            cdf_1(x_values) ** 2
            - 2 * cdf_1(x_values) * cdf_observations(x_values)
            + cdf_observations(x_values),
            x_values,
        )
        skills_CRPS_1.append(1 - CRPS_1 / CRPS_naive)
        CRPS_values_1.append(CRPS_1)

        if model_2_predictions is not None:
            model_2_predictions_resample = np.random.choice(
                model_2_predictions, size=len(observations), replace=True
            )
            cdf_2 = scipy.stats.ecdf(model_2_predictions_resample).cdf.evaluate
            CRPS_2 = np.trapz(
                cdf_2(x_values) ** 2
                - 2 * cdf_2(x_values) * cdf_observations(x_values)
                + cdf_observations(x_values),
                x_values,
            )
            skills_CRPS_2.append(1 - CRPS_2 / CRPS_naive)
            CRPS_values_2.append(CRPS_2)

    cdf_1 = scipy.stats.ecdf(model_1_predictions).cdf.evaluate
    cdf_naive = scipy.stats.ecdf([np.median(observations)]).cdf.evaluate
    cdf_observations = scipy.stats.ecdf(observations).cdf.evaluate

    CRPS_1 = np.trapz(
        cdf_1(x_values) ** 2
        - 2 * cdf_1(x_values) * cdf_observations(x_values)
        + cdf_observations(x_values),
        x_values,
    )

    CRPS_naive = np.trapz(
        cdf_naive(x_values) ** 2
        - 2 * cdf_naive(x_values) * cdf_observations(x_values)
        + cdf_observations(x_values),
        x_values,
    )
    CI_CRPS_1 = [
        np.round(np.percentile(CRPS_values_1, 5), 2),
        np.round(np.percentile(CRPS_values_1, 95), 2),
    ]

    CI_CRPS_skill_1 = [
        np.round(np.percentile(skills_CRPS_1, 5), 2),
        np.round(np.percentile(skills_CRPS_1, 95), 2),
    ]

    # Save the results to a dictionary
    results_dict = {
        "CRPS_1": np.round(CRPS_1, 2),
        "CI(CRPS_1)": CI_CRPS_1,
        "S(CRPS_1)": np.round(1 - CRPS_1 / CRPS_naive, 2),
        "CI(S(CRPS_1))": CI_CRPS_skill_1,
    }

    # When comparing two models, repeat all steps for the second model
    if model_2_predictions is not None:
        cdf_2 = scipy.stats.ecdf(model_2_predictions).cdf.evaluate
        CRPS_2 = np.trapz(
            cdf_2(x_values) ** 2
            - 2 * cdf_2(x_values) * cdf_observations(x_values)
            + cdf_observations(x_values),
            x_values,
        )
        CI_CRPS_2 = [
            np.round(np.percentile(CRPS_values_2, 5), 2),
            np.round(np.percentile(CRPS_values_2, 95), 2),
        ]

        CRPS_ratios = np.array(CRPS_values_1) / np.array(CRPS_values_2)
        CI_CPRS_ratios = [
            np.round(np.percentile(CRPS_ratios, 5), 2),
            np.round(np.percentile(CRPS_ratios, 95), 2),
        ]

        CI_CRPS_skill_2 = [
            np.round(np.percentile(skills_CRPS_2, 5), 2),
            np.round(np.percentile(skills_CRPS_2, 95), 2),
        ]

        # Update the results dictionary with the metrics of the second model
        results_dict.update(
            {
                "CRPS_2": np.round(CRPS_2, 2),
                "CI(CRPS_2)": CI_CRPS_2,
                "S(CRPS_2)": np.round(1 - CRPS_2 / CRPS_naive, 2),
                "CI(S(CRPS_2))": CI_CRPS_skill_2,
                "CRPS_ratio": np.round(CRPS_1 / CRPS_2, 2),
                "CI(CRPS_ratio)": CI_CPRS_ratios,
            }
        )

    return results_dict


def calculate_metrics_comparison(observations, model_1_predictions, model_2_predictions):
    """Calculate all metrics when comparing two models"""
    geomean_observations = scipy.stats.gmean(observations)
    geomean_1 = scipy.stats.gmean(model_1_predictions)
    geomean_2 = scipy.stats.gmean(model_2_predictions)
    GMR_1, CI_GMR_1 = CI_G_LD(
        geomean_observations,
        geomean_1,
        GCV(observations),
        GCV(model_1_predictions),
        len(observations),
        len(model_1_predictions),
        alpha=0.1,
    )
    GMR_2, CI_GMR_2 = CI_G_LD(
        geomean_observations,
        geomean_2,
        GCV(observations),
        GCV(model_2_predictions),
        len(observations),
        len(model_2_predictions),
        alpha=0.1,
    )
    CRPS_results = CRPS_metrics(observations, model_1_predictions, model_2_predictions)
    return GMR_1, CI_GMR_1, GMR_2, CI_GMR_2, CRPS_results


def calculate_metrics_single(observations, model_1_predictions):
    """Calculate all metrics when evaluating a single model."""
    geomean_observations = scipy.stats.gmean(observations)
    geomean_1 = scipy.stats.gmean(model_1_predictions)
    GMR, CI_GMR = CI_G_LD(
        geomean_observations,
        geomean_1,
        GCV(observations),
        GCV(model_1_predictions),
        len(observations),
        len(model_1_predictions),
        alpha=0.1,
    )
    CRPS_results = CRPS_metrics(observations, model_1_predictions)
    return np.round(GMR, 2), CI_GMR, CRPS_results


def plot_comparison(observations, model_1_predictions, model_2_predictions, xlabel):
    """Produce the plots when comparing two models"""
    cdf_1 = scipy.stats.ecdf(model_1_predictions).cdf.evaluate
    cdf_2 = scipy.stats.ecdf(model_2_predictions).cdf.evaluate
    cdf_observations = scipy.stats.ecdf(observations).cdf.evaluate
    max_value = max([max(observations), max(model_1_predictions), max(model_2_predictions)])
    min_value = min([min(observations), min(model_1_predictions), min(model_2_predictions)])
    x_values = np.linspace(0, max_value, 800)
    f1 = gaussian_kde(model_1_predictions, bw_method="silverman")
    f2 = gaussian_kde(model_2_predictions, bw_method="silverman")
    f_observations = gaussian_kde(observations, bw_method="silverman")

    # Make the figure consisting of a plot with the estimated PDFs and two plots with the emperical CDFs
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, dpi=150, figsize=(12, 8))

    ## PDF subplot ##
    ax1.set_ylabel("PDF")
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    # Observations pdf
    ax1.plot(x_values, f_observations(x_values), color="black", label="Observations")
    ax1.fill_between(x_values, f_observations(x_values), color="black", alpha=0.1)
    ax1.hist(observations, density=True, color="black", alpha=0.5, edgecolor="black")

    # Model 1 pdf
    ax1.plot(x_values, f1(x_values), color="blue", label="Model 1")
    ax1.fill_between(x_values, f1(x_values), color="blue", alpha=0.1)

    # Model 2 pdf
    ax1.plot(x_values, f2(x_values), color="red", label="Model 2")
    ax1.fill_between(x_values, f2(x_values), color="red", alpha=0.1)
    ax1.legend()

    ## CDF model 1 subplot ##
    ax2.plot(x_values, cdf_1(x_values), color="blue", label="Model 1")
    ax2.fill_between(
        x_values, cdf_1(x_values), cdf_observations(x_values), color="blue", alpha=0.25
    )
    ax2.set_ylim([0, 1.01])
    ax2.set_ylabel("CDF")
    ax2.plot(x_values, cdf_observations(x_values), color="black", label="Observations")
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)

    ## CDF model 2 subplot ##
    ax3.plot(x_values, cdf_2(x_values), color="red", label="Model 2")
    ax3.fill_between(x_values, cdf_2(x_values), cdf_observations(x_values), color="red", alpha=0.25)
    ax3.plot(x_values, cdf_observations(x_values), color="black", label="Observations")
    ax3.set_ylabel("CDF")
    ax3.set_ylim([0, 1.01])
    ax3.set_xlabel(f"{xlabel}")
    ax3.spines["top"].set_visible(True)
    ax3.spines["right"].set_visible(True)
    fig.tight_layout()
    return fig


def plot_single(observations, predictions, xlabel, fontsize=10):
    """Produce the plots when evaluating a single model."""
    model_1_predictions = predictions
    cdf_1 = scipy.stats.ecdf(model_1_predictions).cdf.evaluate
    # cdf_2 = scipy.stats.ecdf([np.mean(observations)]).cdf.evaluate
    cdf_observations = scipy.stats.ecdf(observations).cdf.evaluate
    max_value = max([max(observations), max(model_1_predictions)])
    min_value = min([min(observations), min(model_1_predictions)])
    x_values = np.linspace(0, max_value, 800)
    f1 = gaussian_kde(model_1_predictions, bw_method="silverman")
    f_observations = gaussian_kde(observations, bw_method="silverman")

    # Make the figure consisting of a plot with the estimated PDFs and two plots with the emperical CDFs
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=150, figsize=(6, 3))

    ## PDF subplot ##
    ax1.set_ylabel("PDF", fontsize=fontsize)
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    # Observations pdf
    ax1.plot(x_values, f_observations(x_values), color="black", label="Observations")
    ax1.fill_between(x_values, f_observations(x_values), color="black", alpha=0.1)
    ax1.hist(observations, density=True, color="black", alpha=0.5, edgecolor="black")

    # Model 1 pdf
    ax1.plot(x_values, f1(x_values), color="blue", label="Model")
    ax1.fill_between(x_values, f1(x_values), color="blue", alpha=0.1)
    ax1.legend(fontsize=fontsize)

    ## CDF subplot ##
    ax2.plot(x_values, cdf_1(x_values), color="blue")
    ax2.fill_between(
        x_values, cdf_1(x_values), cdf_observations(x_values), color="blue", alpha=0.25
    )
    ax2.set_ylim([0, 1.01])
    ax2.set_ylabel("CDF", fontsize=fontsize)
    ax2.plot(x_values, cdf_observations(x_values), color="black")
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.set_xlabel(f"{xlabel}", fontsize=fontsize)
    fig.tight_layout()
    return fig
