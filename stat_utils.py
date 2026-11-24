import numpy as np
from scipy import stats


def ci95(values):
    """
    Half-width of the 95% confidence interval of the mean:
    t * standard error, with the t value of len(values) - 1 degrees of
    freedom (2.78 for five values).
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan

    t_value = stats.t.ppf(0.975, len(values) - 1)
    standard_error = values.std(ddof=1) / np.sqrt(len(values))

    return t_value * standard_error


def holm_correction(p_values):
    """
    Holm correction for multiple comparisons.
    returns the adjusted p-values, in the order of the input
    """
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    num_tests = len(p_values)

    adjusted = np.empty_like(p_values)
    running_max = 0.0

    for rank, idx in enumerate(order):
        # the smallest p-value is multiplied by the number of tests,
        # the next one by one less, and so on
        corrected = (num_tests - rank) * p_values[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(1.0, running_max)

    return adjusted


def cliffs_delta(x, y):
    """
    Cliff's delta of two independent samples:
    (number of pairs with x > y  -  number of pairs with x < y) / all pairs
    """
    x = np.asarray(x)
    y = np.asarray(y)

    num_greater = 0
    num_less = 0
    for x_value in x:
        num_greater += (x_value > y).sum()
        num_less += (x_value < y).sum()

    return (num_greater - num_less) / (len(x) * len(y))


def paired_cliffs_delta(x, y):
    """
    Cliff's delta of paired samples (x[i] and y[i] belong together):
    (number of pairs with x > y  -  number of pairs with x < y) / pairs
    """
    differences = np.asarray(x) - np.asarray(y)

    num_greater = np.sum(differences > 0)
    num_less = np.sum(differences < 0)

    return (num_greater - num_less) / len(differences)


def friedman_test(table):
    """
    Friedman test for a table with one row per unit and one column per
    condition, and Kendall's W as its effect size.

    table: pandas DataFrame without missing values
    returns chi2, p, kendall_w
    """
    columns = [table[c].values for c in table.columns]
    chi2, p = stats.friedmanchisquare(*columns)

    num_units = len(table)
    num_conditions = len(table.columns)
    kendall_w = chi2 / (num_units * (num_conditions - 1))

    return float(chi2), float(p), float(kendall_w)
