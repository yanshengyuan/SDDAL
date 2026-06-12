import os
import re
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.stats import ttest_ind

# =========================================================
# 0) Global display control
# =========================================================
SHOW_METRIC_VALUES_ON_POINTS = True

# =========================================================
# 0.1) External arguments
# =========================================================
parser = argparse.ArgumentParser(
    description="Compute statistics and plot training curves for selected methods under one beam shape."
)

parser.add_argument(
    "--beamshape",
    type=str,
    required=True,
    help="Beam shape name, e.g., hat, chair, rec, ring, gaussian, tear"
)

parser.add_argument(
    "--methods",
    type=str,
    nargs="+",
    required=True,
    help=(
        "List of methods to compare, e.g. "
        "--methods random sddal poolAL GA DE PaPQS"
    )
)

args = parser.parse_args()

beamshape = args.beamshape
methods = args.methods

# =========================================================
# 0.1.1) Check random reference method
# =========================================================
random_candidates = [
    method for method in methods
    if method.lower() == "random"
]

if len(random_candidates) == 0:
    raise ValueError(
        "The method list must include 'random', because p-values are always "
        "computed between each non-random method and random."
    )

if len(random_candidates) > 1:
    raise ValueError(
        "The method list contains duplicated random-like methods. "
        "Please include only one random method."
    )

method_reference = random_candidates[0]

# =========================================================
# 0.2) Display names
# =========================================================
beamshape_display_map = {
    "rec": "RecTophat",
    "ring": "Ring",
    "gaussian": "Gaussian",
    "tear": "Tear",
    "hat": "Tophat",
    "chair": "Chair",
}

method_display_map = {
    "sddal": "SDDAL",
    "SDDAL": "SDDAL",

    "random": "Prior sampling",
    "Random": "Prior sampling",

    "poolAL": "Pool-based AL",
    "poolal": "Pool-based AL",
    "PoolAL": "Pool-based AL",

    "GA": "Genetic Algorithm",
    "ga": "Genetic Algorithm",

    "DE": "Differential Evolution",
    "de": "Differential Evolution",

    "PaPQS": "PaPQS",
    "papqs": "PaPQS",
    "PAPQS": "PaPQS",
}

BEAMSHAPE = beamshape_display_map.get(beamshape, beamshape)
BEAMSHAPE_BOLD = r"$\bf{" + BEAMSHAPE + "}$"

method_display_names = {
    method: method_display_map.get(method, method)
    for method in methods
}

# =========================================================
# 0.3) Plot display settings
# =========================================================
USE_SEED_SPECIFIC_MARKERS_ON_RAW_PLOT = True

SEED_MARKERS = ['o', 's', '^', 'D', 'v']

# =========================================================
# 0.4) Method color settings
# =========================================================
FIXED_METHOD_COLORS = {
    "random": "#000000",
    "Random": "#000000",

    "sddal": "#00C853",
    "SDDAL": "#00C853",

    "poolAL": "#00AEEF",
    "poolal": "#00AEEF",
    "PoolAL": "#00AEEF",

    "GA": "#FFB000",
    "ga": "#FFB000",

    "DE": "#E6007E",
    "de": "#E6007E",

    "PaPQS": "#7B61FF",
    "papqs": "#7B61FF",
    "PAPQS": "#7B61FF",
}

FALLBACK_METHOD_COLORS = [
    "#FF6D00",
    "#00BFA5",
    "#2962FF",
    "#D500F9",
    "#C6FF00",
    "#FF4081",
    "#64DD17",
    "#18FFFF",
    "#FFD600",
    "#AA00FF",
]

method_color_map = {}
used_colors = set()

for method in methods:
    if method in FIXED_METHOD_COLORS:
        color = FIXED_METHOD_COLORS[method]

        if color in used_colors:
            raise ValueError(
                f"Color conflict: method '{method}' wants color '{color}', "
                "but this color has already been used. "
                "Please check whether duplicated or alias method names are included."
            )

        method_color_map[method] = color
        used_colors.add(color)

fallback_color_idx = 0

for method in methods:
    if method in method_color_map:
        continue

    while (
        fallback_color_idx < len(FALLBACK_METHOD_COLORS)
        and FALLBACK_METHOD_COLORS[fallback_color_idx] in used_colors
    ):
        fallback_color_idx += 1

    if fallback_color_idx >= len(FALLBACK_METHOD_COLORS):
        raise ValueError(
            "Not enough unique colors for all selected methods. "
            "Please add more colors to FALLBACK_METHOD_COLORS."
        )

    color = FALLBACK_METHOD_COLORS[fallback_color_idx]
    method_color_map[method] = color
    used_colors.add(color)
    fallback_color_idx += 1

# =========================================================
# 0.5) Font and layout settings
# =========================================================
NUMERIC_FONT_SCALE = 1.3

METRIC_VALUE_FONT_SIZE = round(13 * NUMERIC_FONT_SCALE)
P_VALUE_TABLE_FONT_SIZE = round(12 * NUMERIC_FONT_SCALE)
TICK_FONT_SIZE = round(14 * NUMERIC_FONT_SCALE)

AXIS_LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 15

METRIC_VALUE_VERTICAL_OFFSET_RATIO = 0.025
MIN_LABEL_SEPARATION_RATIO = 0.040

PRINT_P_VALUE_TABLE = True

# =========================================================
# 0.6) P-value table layout settings
# =========================================================
TABLE_FONT_SCALE_FOR_MANY_METHODS = True
P_VALUE_TABLE_TOP_MARGIN = 0.05
P_VALUE_TABLE_BOTTOM_MARGIN = 0.05

# =========================================================
# 1) Path settings
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

method_dirs = {}

for method in methods:
    method_dir = os.path.join(
        base_dir,
        "training_curve_" + method + "_" + beamshape
    )

    if not os.path.isdir(method_dir):
        raise FileNotFoundError(
            f"Folder not found for method '{method}': {method_dir}"
        )

    method_dirs[method] = method_dir

# =========================================================
# 2) Basic configuration
# =========================================================
seed_names = [
    beamshape + "_1",
    beamshape + "_2",
    beamshape + "_3",
    beamshape + "_4",
    beamshape + "_5",
]

sample_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

seed_marker_map = {
    seed_name: SEED_MARKERS[i % len(SEED_MARKERS)]
    for i, seed_name in enumerate(seed_names)
}

method_label_map = {
    method: method_display_names[method]
    for method in methods
}

method_string_for_save = "_vs_".join(methods)

metric_info = {
    "MAE": {
        "ylabel": "MAE",
        "raw_save_name": f"all_curves_MAE_{beamshape}_{method_string_for_save}.png",
        "mean_save_name": f"mean_std_curves_MAE_{beamshape}_{method_string_for_save}.png",
    },
    "SSIM": {
        "ylabel": "SSIM",
        "raw_save_name": f"all_curves_SSIM_{beamshape}_{method_string_for_save}.png",
        "mean_save_name": f"mean_std_curves_SSIM_{beamshape}_{method_string_for_save}.png",
    },
    "FRCM": {
        "ylabel": "FRCM",
        "raw_save_name": f"all_curves_FRCM_{beamshape}_{method_string_for_save}.png",
        "mean_save_name": f"mean_std_curves_FRCM_{beamshape}_{method_string_for_save}.png",
    }
}

# =========================================================
# 3) Regular expressions
# =========================================================
pattern_dict = {
    "MAE": re.compile(r"Mean\s+MAE\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "SSIM": re.compile(r"Mean\s+SSIM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "FRCM": re.compile(r"Mean\s+FRCM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}

# =========================================================
# 4) Data structure
# =========================================================
all_results = {
    method: {}
    for method in methods
}

invalid_eval_paths = []
missing_eval_paths = []

pvalue_results = {
    metric_name: {}
    for metric_name in metric_info.keys()
}


def is_finite_number(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def extract_metric(text, metric_name):
    match = pattern_dict[metric_name].search(text)
    if match is None:
        return None

    try:
        value = float(match.group(1))
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def format_p_value(p):
    """
    Format p-value with one significant digit.
    """
    if p is None or not isinstance(p, (int, float)) or not math.isfinite(p):
        return "nan"

    if p == 0:
        return "0"

    if abs(p) < 1e-3:
        return f"{p:.0e}"

    return f"{p:.1g}"


def format_metric_value(v, metric_name):
    """
    Format metric value with two significant digits.
    """
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "nan"
    return f"{v:.2g}"


def safe_mean(values):
    valid = [v for v in values if is_finite_number(v)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid))


def compute_metric_exists(metric_name):
    for method in all_results:
        for seed_name in all_results[method]:
            values = all_results[method][seed_name][metric_name]
            if any(is_finite_number(v) for v in values):
                return True
    return False


print("=" * 80)
print("Scanning evaluation.txt files...")
print(f"Beamshape argument: {beamshape}")
print(f"Beamshape display:  {BEAMSHAPE}")
print(f"Selected methods:   {methods}")
print(f"Reference method for p-values: {method_reference}")
print("P-values are computed between each non-random method and random.")
print("=" * 80)

if USE_SEED_SPECIFIC_MARKERS_ON_RAW_PLOT:
    print("\nSeed-marker mapping used in raw plots:")
    for seed_name in seed_names:
        print(f"  {seed_name} -> {seed_marker_map[seed_name]}")

print("\nMethod-color mapping:")
for method in methods:
    print(f"  {method_label_map[method]} ({method}) -> {method_color_map[method]}")

# =========================================================
# 5.1) Scan evaluation files
# =========================================================
for method in methods:
    method_dir = method_dirs[method]
    method_label = method_label_map[method]

    print(f"\nMethod: {method_label}")
    print(f"Directory: {method_dir}")

    all_results[method] = {}

    for seed_name in seed_names:
        seed_dir = os.path.join(method_dir, seed_name)

        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"Seed folder not found: {seed_dir}")

        all_results[method][seed_name] = {
            "MAE": [],
            "SSIM": [],
            "FRCM": [],
        }

        print(f"  Reading {seed_name} ...")

        for n in sample_sizes:
            exp_name = f"{seed_name}_{n}"
            exp_dir = os.path.join(seed_dir, exp_name)
            eval_path = os.path.join(exp_dir, "evaluation.txt")

            if not os.path.isfile(eval_path):
                missing_eval_paths.append(eval_path)
                print(f"    [SKIP] missing evaluation.txt: {eval_path}")
                all_results[method][seed_name]["MAE"].append(None)
                all_results[method][seed_name]["SSIM"].append(None)
                all_results[method][seed_name]["FRCM"].append(None)
                continue

            with open(eval_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            mae_value = extract_metric(text, "MAE")
            ssim_value = extract_metric(text, "SSIM")
            frcm_value = extract_metric(text, "FRCM")

            if mae_value is None or ssim_value is None:
                invalid_eval_paths.append(eval_path)
                print(f"    [SKIP] invalid data in: {eval_path}")
                all_results[method][seed_name]["MAE"].append(None)
                all_results[method][seed_name]["SSIM"].append(None)
                all_results[method][seed_name]["FRCM"].append(None)
                continue

            all_results[method][seed_name]["MAE"].append(mae_value)
            all_results[method][seed_name]["SSIM"].append(ssim_value)
            all_results[method][seed_name]["FRCM"].append(frcm_value)

            print(
                f"    {exp_name}: "
                f"MAE={mae_value:.12f}, "
                f"SSIM={ssim_value:.12f}, "
                f"FRCM={'None' if frcm_value is None else f'{frcm_value:.12f}'}"
            )

print("\nFinished scanning all evaluation files.")

if missing_eval_paths:
    print("\n" + "=" * 80)
    print("Missing evaluation.txt files:")
    print("=" * 80)
    for p in missing_eval_paths:
        print(p)

if invalid_eval_paths:
    print("\n" + "=" * 80)
    print("evaluation.txt files with invalid / unreadable metric data:")
    print("=" * 80)
    for p in invalid_eval_paths:
        print(p)

# =========================================================
# 6) Compute raw p-values
# =========================================================
for metric_name in metric_info.keys():
    metric_exists = compute_metric_exists(metric_name)

    if not metric_exists:
        print(f"\nSkip p-value computation for {metric_name}: no valid data found.")

        for method in methods:
            if method != method_reference:
                pvalue_results[metric_name][method] = [np.nan] * len(sample_sizes)

        continue

    for method in methods:
        if method == method_reference:
            continue

        raw_p_values = []

        for idx, n in enumerate(sample_sizes):
            group_random = []
            group_method = []

            for seed_name in seed_names:
                v_random = all_results[method_reference][seed_name][metric_name][idx]
                v_method = all_results[method][seed_name][metric_name][idx]

                if is_finite_number(v_random):
                    group_random.append(v_random)

                if is_finite_number(v_method):
                    group_method.append(v_method)

            if len(group_random) >= 2 and len(group_method) >= 2:
                _, p_value = ttest_ind(
                    group_method,
                    group_random,
                    equal_var=False,
                    nan_policy="omit"
                )

                if p_value is None or not math.isfinite(p_value):
                    p_value = np.nan
            else:
                p_value = np.nan

            raw_p_values.append(p_value)

        pvalue_results[metric_name][method] = raw_p_values

# =========================================================
# 7) Print p-value result table
# =========================================================
if PRINT_P_VALUE_TABLE and len(methods) >= 2:
    print("\n" + "=" * 100)
    print("P-VALUE RESULTS (Welch's t-test)")
    print(f"Reference method: {method_reference}")
    print("Each non-random method is compared against random.")
    print("=" * 100)

    for metric_name in metric_info.keys():
        print(f"\nMetric: {metric_name}")
        print("-" * 100)

        random_mean_name = method_label_map[method_reference] + "_mean"
        header = f"{'sample_size':>12} | {random_mean_name:>20}"

        for method in methods:
            if method == method_reference:
                continue

            method_mean_name = method_label_map[method] + "_mean"
            header += f" | {method_mean_name:>20} | {'p_vs_random':>12}"

        print(header)
        print("-" * 100)

        for idx, n in enumerate(sample_sizes):
            random_values = [
                all_results[method_reference][seed_name][metric_name][idx]
                for seed_name in seed_names
            ]

            random_mean = safe_mean(random_values)
            random_mean_str = "nan" if not math.isfinite(random_mean) else f"{random_mean:.6f}"

            row = f"{n:12d} | {random_mean_str:>20}"

            for method in methods:
                if method == method_reference:
                    continue

                method_values = [
                    all_results[method][seed_name][metric_name][idx]
                    for seed_name in seed_names
                ]

                method_mean = safe_mean(method_values)
                raw_p = pvalue_results[metric_name][method][idx]

                method_mean_str = "nan" if not math.isfinite(method_mean) else f"{method_mean:.6f}"
                raw_p_str = format_p_value(raw_p)

                row += f" | {method_mean_str:>20} | {raw_p_str:>12}"

            print(row)

# =========================================================
# 8) Plot all raw curves
# =========================================================
for metric_name, info in metric_info.items():
    metric_exists = compute_metric_exists(metric_name)

    if not metric_exists:
        print(f"\nSkip {metric_name}: no valid '{metric_name}' found in any evaluation.txt")
        continue

    fig = plt.figure(figsize=(14, 6))

    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[4.8, 1.4],
        wspace=0.05
    )

    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")

    for method in methods:
        color = method_color_map[method]

        for seed_idx, seed_name in enumerate(seed_names):
            y = all_results[method][seed_name][metric_name]

            x_valid = [
                x for x, v in zip(sample_sizes, y)
                if is_finite_number(v)
            ]

            y_valid = [
                v for v in y
                if is_finite_number(v)
            ]

            if len(x_valid) == 0:
                print(
                    f"Warning: no valid {metric_name} points in "
                    f"{method_label_map[method]}/{seed_name}, skip this curve."
                )
                continue

            marker_style = seed_marker_map[seed_name] if USE_SEED_SPECIFIC_MARKERS_ON_RAW_PLOT else "o"

            ax.plot(
                x_valid,
                y_valid,
                marker=marker_style,
                markersize=5,
                linewidth=2.8,
                linestyle="-",
                color=color,
                alpha=1.0,
                label="_nolegend_"
            )

    ax.set_xlabel("Number of training samples", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(info["ylabel"], fontsize=AXIS_LABEL_FONT_SIZE)

    ax.set_title(
        f"{metric_name} vs. Number of training samples of {BEAMSHAPE_BOLD}",
        fontsize=TITLE_FONT_SIZE
    )

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    method_handles = []

    for method in methods:
        method_handles.append(
            Line2D(
                [0],
                [0],
                color=method_color_map[method],
                marker="o",
                linestyle="-",
                linewidth=3.0,
                markersize=6,
                label=method_label_map[method]
            )
        )

    method_legend = legend_ax.legend(
        handles=method_handles,
        title="Methods",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.00),
        borderaxespad=0.0,
        frameon=True
    )

    legend_ax.add_artist(method_legend)

    seed_handles = []

    for seed_idx, seed_name in enumerate(seed_names):
        marker_style = seed_marker_map[seed_name] if USE_SEED_SPECIFIC_MARKERS_ON_RAW_PLOT else "o"

        seed_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=marker_style,
                linestyle="None",
                markersize=7,
                label=f"seed={seed_idx + 1}"
            )
        )

    legend_ax.legend(
        handles=seed_handles,
        title="Seeds",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.58),
        borderaxespad=0.0,
        frameon=True
    )

    save_path = os.path.join(base_dir, info["raw_save_name"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.show()
    plt.close()

    print(f"Saved raw curve figure: {save_path}")

# =========================================================
# 9) Plot mean curves with std regions
# =========================================================
for metric_name, info in metric_info.items():
    metric_exists = compute_metric_exists(metric_name)

    if not metric_exists:
        continue

    comparison_methods = [
        method for method in methods
        if method != method_reference
    ]

    table_rows = 1 + len(comparison_methods)

    if len(comparison_methods) > 0:
        fig_height = 7.0 + 0.38 * max(0, table_rows - 2)
        table_height_ratio = 0.95 + 0.28 * max(0, table_rows - 2)
    else:
        fig_height = 7.0
        table_height_ratio = 0.55

    fig = plt.figure(figsize=(10, fig_height))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[4.0, table_height_ratio],
        hspace=0.35
    )

    ax = fig.add_subplot(gs[0, 0])
    table_ax = fig.add_subplot(gs[1, 0])
    table_ax.axis("off")

    stored_mean_curves = {}
    stored_std_curves = {}

    for method in methods:
        method_label = method_label_map[method]
        color = method_color_map[method]

        mean_values = []
        std_values = []

        for idx, n in enumerate(sample_sizes):
            point_values = []

            for seed_name in seed_names:
                v = all_results[method][seed_name][metric_name][idx]

                if is_finite_number(v):
                    point_values.append(v)

            if len(point_values) == 0:
                mean_values.append(np.nan)
                std_values.append(np.nan)
            else:
                point_values = np.array(point_values, dtype=float)
                mean_values.append(float(np.mean(point_values)))

                if len(point_values) >= 2:
                    std_values.append(float(np.std(point_values, ddof=1)))
                else:
                    std_values.append(0.0)

        x_arr = np.array(sample_sizes, dtype=float)
        mean_arr = np.array(mean_values, dtype=float)
        std_arr = np.array(std_values, dtype=float)

        stored_mean_curves[method] = mean_arr.copy()
        stored_std_curves[method] = std_arr.copy()

        valid_mask = np.isfinite(mean_arr)

        if not np.any(valid_mask):
            print(
                f"Warning: no valid mean/std points for "
                f"{metric_name} in {method_label}, skip."
            )
            continue

        lower_arr = mean_arr - std_arr
        upper_arr = mean_arr + std_arr

        ax.fill_between(
            x_arr,
            lower_arr,
            upper_arr,
            where=valid_mask,
            interpolate=True,
            color=color,
            alpha=0.07,
        )

        ax.plot(
            x_arr[valid_mask],
            mean_arr[valid_mask],
            marker="o",
            markersize=5,
            linewidth=3.2,
            linestyle="-",
            color=color,
            alpha=1.0,
            label=method_label,
        )

    ax.set_xlabel(
        "Number of training samples",
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=10
    )

    ax.set_ylabel(info["ylabel"], fontsize=AXIS_LABEL_FONT_SIZE)

    ax.set_title(
        f"{metric_name} mean curve with std region of {BEAMSHAPE_BOLD}",
        fontsize=TITLE_FONT_SIZE
    )

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="best")
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    fig.canvas.draw()

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    base_text_offset = METRIC_VALUE_VERTICAL_OFFSET_RATIO * y_range
    min_label_separation = MIN_LABEL_SEPARATION_RATIO * y_range

    # -----------------------------------------------------
    # 9.1) Optional metric-value annotations
    #
    # New rule for each sample-size column:
    #   top point    -> label above its curve
    #   bottom point -> label below its curve
    #   all middle points -> label above their curves
    #
    # For all labels placed above, a small upward separation is applied
    # to reduce overlaps while keeping them above their corresponding curves.
    # -----------------------------------------------------
    if SHOW_METRIC_VALUES_ON_POINTS:
        text_y_values = []

        for idx, n in enumerate(sample_sizes):
            valid_method_values = []

            for method in methods:
                mean_arr = stored_mean_curves.get(
                    method,
                    np.full(len(sample_sizes), np.nan)
                )

                if idx < len(mean_arr):
                    y_value = mean_arr[idx]
                else:
                    y_value = np.nan

                if math.isfinite(y_value):
                    valid_method_values.append((method, y_value))

            if len(valid_method_values) == 0:
                continue

            if len(valid_method_values) == 1:
                method, y_value = valid_method_values[0]
                text_y = y_value + base_text_offset

                ax.text(
                    n,
                    text_y,
                    format_metric_value(y_value, metric_name),
                    ha="center",
                    va="bottom",
                    fontsize=METRIC_VALUE_FONT_SIZE,
                    color=method_color_map[method]
                )

                text_y_values.append(text_y)
                continue

            # Determine top and bottom points in this x-column
            sorted_by_y = sorted(
                valid_method_values,
                key=lambda item: item[1]
            )

            bottom_method, bottom_y = sorted_by_y[0]
            top_method, top_y = sorted_by_y[-1]

            # Put the bottom point label below the curve
            bottom_text_y = bottom_y - base_text_offset

            ax.text(
                n,
                bottom_text_y,
                format_metric_value(bottom_y, metric_name),
                ha="center",
                va="top",
                fontsize=METRIC_VALUE_FONT_SIZE,
                color=method_color_map[bottom_method]
            )

            text_y_values.append(bottom_text_y)

            # Put top and middle point labels above their curves
            above_label_items = []

            for method, y_value in valid_method_values:
                if method == bottom_method and y_value == bottom_y:
                    continue

                initial_text_y = y_value + base_text_offset

                above_label_items.append(
                    {
                        "method": method,
                        "value": y_value,
                        "text_y": initial_text_y,
                    }
                )

            above_label_items = sorted(
                above_label_items,
                key=lambda item: item["text_y"]
            )

            # Enforce minimum vertical separation among above labels
            for k in range(1, len(above_label_items)):
                previous_text_y = above_label_items[k - 1]["text_y"]
                current_text_y = above_label_items[k]["text_y"]

                if current_text_y - previous_text_y < min_label_separation:
                    above_label_items[k]["text_y"] = previous_text_y + min_label_separation

            for item in above_label_items:
                method = item["method"]
                y_value = item["value"]
                text_y = item["text_y"]

                ax.text(
                    n,
                    text_y,
                    format_metric_value(y_value, metric_name),
                    ha="center",
                    va="bottom",
                    fontsize=METRIC_VALUE_FONT_SIZE,
                    color=method_color_map[method]
                )

                text_y_values.append(text_y)

        if len(text_y_values) > 0:
            new_y_min = min(y_min, min(text_y_values) - 2.0 * base_text_offset)
            new_y_max = max(y_max, max(text_y_values) + 2.0 * base_text_offset)
            ax.set_ylim(new_y_min, new_y_max)

    # -----------------------------------------------------
    # 9.2) Independent p-value table
    # -----------------------------------------------------
    if len(comparison_methods) >= 1:
        table_data = [
            ["Samples"] + [str(n) for n in sample_sizes]
        ]

        for method in comparison_methods:
            pvals_to_show = pvalue_results[metric_name][method]

            pvalue_texts = [
                format_p_value(pvals_to_show[idx])
                for idx in range(len(sample_sizes))
            ]

            row_name = method_label_map[method]
            table_data.append([row_name] + pvalue_texts)

        n_cols = len(table_data[0])

        column_weights = []

        for col_idx in range(n_cols):
            col_texts = [
                str(table_data[row_idx][col_idx])
                for row_idx in range(len(table_data))
            ]

            max_len = max(len(txt) for txt in col_texts)
            weight = max_len + 1.5

            if col_idx == 0:
                weight += 1.8

            column_weights.append(weight)

        weight_sum = sum(column_weights)

        col_widths = [
            w / weight_sum
            for w in column_weights
        ]

        current_table_font_size = P_VALUE_TABLE_FONT_SIZE

        if TABLE_FONT_SCALE_FOR_MANY_METHODS and len(table_data) >= 4:
            current_table_font_size = max(
                10,
                P_VALUE_TABLE_FONT_SIZE - 1 * (len(table_data) - 3)
            )

        pvalue_table = table_ax.table(
            cellText=table_data,
            cellLoc="center",
            loc="center",
            colWidths=col_widths,
            bbox=[
                0.0,
                P_VALUE_TABLE_BOTTOM_MARGIN,
                1.0,
                1.0 - P_VALUE_TABLE_TOP_MARGIN - P_VALUE_TABLE_BOTTOM_MARGIN
            ]
        )

        pvalue_table.auto_set_font_size(False)
        pvalue_table.set_fontsize(current_table_font_size)

        n_rows = len(table_data)

        for (r, c), cell in pvalue_table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("black")
            cell.set_text_props(
                color="black",
                fontsize=current_table_font_size
            )

            if r == 0:
                cell.visible_edges = "TB"
                cell.set_linewidth(1.2)
                cell.set_text_props(
                    color="black",
                    fontsize=current_table_font_size
                )

            elif r == n_rows - 1:
                cell.visible_edges = "B"
                cell.set_linewidth(1.0)

                if c == 0:
                    cell.set_text_props(
                        color="black",
                        fontsize=current_table_font_size
                    )

            else:
                cell.visible_edges = ""
                cell.set_linewidth(0.0)

        pvalue_table.scale(1.0, 1.15)

    save_path = os.path.join(base_dir, info["mean_save_name"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.show()
    plt.close()

    print(f"Saved mean±std figure: {save_path}")

print("\nAll figures have been generated successfully.")