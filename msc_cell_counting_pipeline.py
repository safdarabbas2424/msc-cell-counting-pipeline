import os
import re
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_FOLDER = Path("/Users/safdarabbas/Desktop/Safdar/cell_counting")
IMAGE_FOLDER = PROJECT_FOLDER / "images"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs_95_plus"

DETECTED_FOLDER = OUTPUT_FOLDER / "detected_images"
BINARY_FOLDER = OUTPUT_FOLDER / "binary_masks"
CORRECTED_FOLDER = OUTPUT_FOLDER / "corrected_images"
FIGURE_FOLDER = OUTPUT_FOLDER / "report_figures"

for folder in [
    OUTPUT_FOLDER,
    DETECTED_FOLDER,
    BINARY_FOLDER,
    CORRECTED_FOLDER,
    FIGURE_FOLDER
]:
    folder.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# ============================================================
# 2. PARAMETERS
# These are saved later so the report can explain them clearly
# ============================================================

PARAMS = {
    "mask_margin_px": 60,
    "background_sigma": 45,
    "morph_kernel_size": 3,
    "min_area": 40,
    "max_area": 3000,
    "min_box_size": 5,
    "min_aspect_ratio": 1.2,
    "strong_aspect_ratio": 1.5,
    "no_growth_area_threshold": 20000,
    "low_confidence_area_threshold": 50000,
    "no_growth_count_threshold": 60
}

PARAMETER_REASONS = [
    ["mask_margin_px", PARAMS["mask_margin_px"], "Removes microscope border and edge artefacts."],
    ["background_sigma", PARAMS["background_sigma"], "Large blur estimates uneven illumination."],
    ["morph_kernel_size", PARAMS["morph_kernel_size"], "Small kernel removes noise without destroying cell shapes."],
    ["min_area", PARAMS["min_area"], "Removes tiny noise specks."],
    ["max_area", PARAMS["max_area"], "Removes very large merged debris or artefacts."],
    ["min_box_size", PARAMS["min_box_size"], "Removes very thin or tiny detections."],
    ["min_aspect_ratio", PARAMS["min_aspect_ratio"], "Keeps elongated MSC-like structures."],
    ["no_growth_area_threshold", PARAMS["no_growth_area_threshold"], "Flags very weak segmented signal as no growth."],
    ["low_confidence_area_threshold", PARAMS["low_confidence_area_threshold"], "Flags sparse/ambiguous signal for review."]
]

# ============================================================
# 3. METADATA PARSING
# Handles AD-G5-t04.tif, UC-H7-t34.tif, noGrowth1.tif
# ============================================================

def parse_metadata(filename):
    stem = Path(filename).stem

    if "nogrowth" in stem.lower() or "no_growth" in stem.lower():
        return {
            "filename": filename,
            "cell_line": "Unknown",
            "well": "Unknown",
            "timepoint": "Unknown",
            "image_type": "no_growth_control"
        }

    match = re.search(
        r"(?P<cell_line>AD|BM|UC)[-_](?P<well>[A-H][0-9]{1,2})[-_]t(?P<timepoint>[0-9]{1,2})",
        stem,
        re.IGNORECASE
    )

    if match:
        return {
            "filename": filename,
            "cell_line": match.group("cell_line").upper(),
            "well": match.group("well").upper(),
            "timepoint": int(match.group("timepoint")),
            "image_type": "growth_series"
        }

    return {
        "filename": filename,
        "cell_line": "Unknown",
        "well": "Unknown",
        "timepoint": "Unknown",
        "image_type": "unknown"
    }

# ============================================================
# 4. IMAGE PROCESSING FUNCTIONS
# ============================================================

def create_circular_mask(gray):
    h, w = gray.shape
    mask = np.zeros_like(gray, dtype=np.uint8)

    radius = max(10, min(h, w) // 2 - PARAMS["mask_margin_px"])
    cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)

    return mask


def background_correct(gray, mask):
    gray_masked = cv2.bitwise_and(gray, mask)

    background = cv2.GaussianBlur(
        gray_masked,
        (0, 0),
        PARAMS["background_sigma"]
    )

    corrected = cv2.absdiff(gray_masked, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    corrected = cv2.bitwise_and(corrected, mask)

    return corrected


def segment_cells(corrected, mask):
    _, binary = cv2.threshold(
        corrected,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.bitwise_and(binary, mask)

    kernel = np.ones(
        (PARAMS["morph_kernel_size"], PARAMS["morph_kernel_size"]),
        np.uint8
    )

    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary_clean


def extract_components(binary, mask):
    h, w = binary.shape

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    detections = []
    object_areas = []
    elongated_count = 0

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        box_w = int(stats[i, cv2.CC_STAT_WIDTH])
        box_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]

        if area < PARAMS["min_area"] or area > PARAMS["max_area"]:
            continue

        if box_w < PARAMS["min_box_size"] or box_h < PARAMS["min_box_size"]:
            continue

        if not (0 <= cx < w and 0 <= cy < h):
            continue

        if mask[int(cy), int(cx)] == 0:
            continue

        aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-5)

        if aspect_ratio < PARAMS["min_aspect_ratio"]:
            continue

        if aspect_ratio >= PARAMS["strong_aspect_ratio"]:
            elongated_count += 1

        detections.append((x, y, box_w, box_h, area, aspect_ratio))
        object_areas.append(area)

    raw_count = len(detections)
    total_area = int(np.sum(object_areas)) if len(object_areas) > 0 else 0
    mean_area = float(np.mean(object_areas)) if len(object_areas) > 0 else 0.0
    median_area = float(np.median(object_areas)) if len(object_areas) > 0 else 0.0

    features = {
        "raw_count": raw_count,
        "elongated_count": elongated_count,
        "total_area": total_area,
        "mean_object_area": round(mean_area, 2),
        "median_object_area": round(median_area, 2)
    }

    return detections, features


def classify_status(features):
    raw_count = features["raw_count"]
    total_area = features["total_area"]
    elongated_count = features["elongated_count"]

    if (
        total_area < PARAMS["no_growth_area_threshold"]
        and raw_count < PARAMS["no_growth_count_threshold"]
    ):
        return {
            "status": "No growth detected",
            "final_count": 0,
            "confidence_score": 0.90,
            "notes": "Very low segmented area and few MSC-like detections."
        }

    if total_area < PARAMS["low_confidence_area_threshold"]:
        confidence = 0.35 + (total_area / PARAMS["low_confidence_area_threshold"]) * 0.30
        confidence = min(confidence, 0.65)

        return {
            "status": "Low-confidence growth",
            "final_count": raw_count,
            "confidence_score": round(confidence, 3),
            "notes": "Weak or sparse signal; visual review recommended."
        }

    elongation_fraction = elongated_count / max(raw_count, 1)
    confidence = min(0.95, 0.70 + 0.20 * elongation_fraction)

    return {
        "status": "Growth detected",
        "final_count": raw_count,
        "confidence_score": round(confidence, 3),
        "notes": "Clear segmented signal inside field of view."
    }


def annotate_image(image, detections, status, count):
    output = image.copy()

    if status == "No growth detected":
        color = (0, 0, 255)
    elif status == "Low-confidence growth":
        color = (0, 220, 220)
    else:
        color = (0, 180, 0)

    for x, y, box_w, box_h, area, aspect_ratio in detections:
        cv2.rectangle(output, (x, y), (x + box_w, y + box_h), color, 1)

    cv2.putText(
        output,
        f"{status} | Count: {count}",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
        cv2.LINE_AA
    )

    return output


def count_cells(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = create_circular_mask(gray)
    corrected = background_correct(gray, mask)
    binary = segment_cells(corrected, mask)

    detections, features = extract_components(binary, mask)
    status_info = classify_status(features)

    final_count = status_info["final_count"]
    status = status_info["status"]

    annotated = annotate_image(image, detections, status, final_count)

    megapixels = (gray.shape[0] * gray.shape[1]) / 1_000_000
    count_density = final_count / max(megapixels, 1e-6)

    result = {
        "cell_count": int(final_count),
        "status": status,
        "confidence_score": status_info["confidence_score"],
        "raw_count": int(features["raw_count"]),
        "elongated_count": int(features["elongated_count"]),
        "total_area": int(features["total_area"]),
        "mean_object_area": features["mean_object_area"],
        "median_object_area": features["median_object_area"],
        "count_density_per_megapixel": round(count_density, 2),
        "notes": status_info["notes"]
    }

    images = {
        "original": image,
        "corrected": corrected,
        "binary": binary,
        "annotated": annotated
    }

    return result, images

# ============================================================
# 5. LOAD AND PROCESS ALL IMAGES
# ============================================================

image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(VALID_EXTENSIONS)
])

print("Total images found:", len(image_files))

if len(image_files) == 0:
    raise ValueError(f"No images found in: {IMAGE_FOLDER}")

results = []

for filename in image_files:
    image_path = IMAGE_FOLDER / filename

    metadata = parse_metadata(filename)
    count_result, images = count_cells(image_path)

    row = {
        "filename": filename,
        "cell_line": metadata["cell_line"],
        "well": metadata["well"],
        "timepoint": metadata["timepoint"],
        "image_type": metadata["image_type"],
        **count_result
    }

    results.append(row)

    stem = Path(filename).stem

    cv2.imwrite(str(DETECTED_FOLDER / f"{stem}_detected.png"), images["annotated"])
    cv2.imwrite(str(BINARY_FOLDER / f"{stem}_binary.png"), images["binary"])
    cv2.imwrite(str(CORRECTED_FOLDER / f"{stem}_corrected.png"), images["corrected"])

print("Pipeline complete.")

# ============================================================
# 6. SAVE RESULTS CSV
# ============================================================

results_csv = OUTPUT_FOLDER / "cell_count_results_95_plus.csv"

fieldnames = [
    "filename",
    "cell_line",
    "well",
    "timepoint",
    "image_type",
    "cell_count",
    "status",
    "confidence_score",
    "raw_count",
    "elongated_count",
    "total_area",
    "mean_object_area",
    "median_object_area",
    "count_density_per_megapixel",
    "notes"
]

with open(results_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("Results CSV saved:", results_csv)

# ============================================================
# 7. SAVE PARAMETER TABLE
# ============================================================

parameter_csv = OUTPUT_FOLDER / "pipeline_parameter_justification.csv"

with open(parameter_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value", "reason"])
    writer.writerows(PARAMETER_REASONS)

print("Parameter table saved:", parameter_csv)

# ============================================================
# 8. HELPER FUNCTIONS FOR PLOTS
# ============================================================

def save_plot(filename):
    path = FIGURE_FOLDER / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", path)


def get_growth_results(results):
    growth = []

    for row in results:
        if row["timepoint"] != "Unknown":
            growth.append(row)

    return growth


growth_results = get_growth_results(results)

# ============================================================
# 9. GRAPH 1: CELL COUNT OVER TIME
# ============================================================

plt.figure(figsize=(10, 6))

series_keys = sorted(set(
    (r["cell_line"], r["well"]) for r in growth_results
))

for cell_line, well in series_keys:
    series = [
        r for r in growth_results
        if r["cell_line"] == cell_line and r["well"] == well
    ]

    series = sorted(series, key=lambda x: x["timepoint"])

    times = [r["timepoint"] for r in series]
    counts = [r["cell_count"] for r in series]

    plt.plot(times, counts, marker="o", label=f"{cell_line}-{well}")

plt.xlabel("Timepoint (hours)")
plt.ylabel("Detected cell count")
plt.title("Detected Cell Count Over Time")
plt.grid(True, alpha=0.35)
plt.legend(fontsize=8)

save_plot("01_cell_count_over_time.png")

# ============================================================
# 10. GRAPH 2: SEGMENTED AREA OVER TIME
# ============================================================

plt.figure(figsize=(10, 6))

for cell_line, well in series_keys:
    series = [
        r for r in growth_results
        if r["cell_line"] == cell_line and r["well"] == well
    ]

    series = sorted(series, key=lambda x: x["timepoint"])

    times = [r["timepoint"] for r in series]
    areas = [r["total_area"] for r in series]

    plt.plot(times, areas, marker="o", label=f"{cell_line}-{well}")

plt.xlabel("Timepoint (hours)")
plt.ylabel("Total segmented area")
plt.title("Total Segmented Area Over Time")
plt.grid(True, alpha=0.35)
plt.legend(fontsize=8)

save_plot("02_segmented_area_over_time.png")

# ============================================================
# 11. GRAPH 3: AVERAGE CELL COUNT BY CELL LINE
# ============================================================

cell_lines = sorted(set(r["cell_line"] for r in growth_results))

average_counts = []
std_counts = []

for cell_line in cell_lines:
    values = [
        r["cell_count"] for r in growth_results
        if r["cell_line"] == cell_line
    ]

    average_counts.append(float(np.mean(values)))
    std_counts.append(float(np.std(values)))

plt.figure(figsize=(7, 5))
plt.bar(cell_lines, average_counts, yerr=std_counts, capsize=5)
plt.xlabel("Cell line")
plt.ylabel("Mean detected cell count")
plt.title("Average Detected Cell Count by Cell Line")

save_plot("03_average_count_by_cell_line.png")

# ============================================================
# 12. GRAPH 4: STATUS DISTRIBUTION
# ============================================================

status_counts = {}

for row in results:
    status = row["status"]
    status_counts[status] = status_counts.get(status, 0) + 1

plt.figure(figsize=(8, 5))
plt.bar(status_counts.keys(), status_counts.values())
plt.xlabel("Status")
plt.ylabel("Number of images")
plt.title("Image Classification Status Summary")
plt.xticks(rotation=20, ha="right")

save_plot("04_status_distribution.png")

# ============================================================
# 13. GRAPH 5: NO-GROWTH CONTROL RESULTS
# ============================================================

control_results = [
    r for r in results
    if r["image_type"] == "no_growth_control"
]

if len(control_results) > 0:
    names = [Path(r["filename"]).stem for r in control_results]
    counts = [r["cell_count"] for r in control_results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, counts)
    plt.xlabel("No-growth control image")
    plt.ylabel("Detected cell count")
    plt.title("No-growth Control Results")
    plt.xticks(rotation=25, ha="right")

    save_plot("05_no_growth_control_counts.png")
else:
    print("No no-growth controls found.")

# ============================================================
# 14. GRAPH 6: INTERNAL QC - AREA VS COUNT
# ============================================================

if len(growth_results) > 1:
    areas = np.array([r["total_area"] for r in growth_results], dtype=float)
    counts = np.array([r["cell_count"] for r in growth_results], dtype=float)

    if np.std(areas) > 0 and np.std(counts) > 0:
        correlation = np.corrcoef(areas, counts)[0, 1]
    else:
        correlation = np.nan

    plt.figure(figsize=(7, 5))
    plt.scatter(areas, counts)
    plt.xlabel("Total segmented area")
    plt.ylabel("Detected cell count")
    plt.title(f"Internal QC: Area vs Count Correlation (r={correlation:.2f})")

    save_plot("06_area_vs_count_qc.png")
else:
    correlation = np.nan
    print("Not enough growth images for area-count QC.")

# ============================================================
# 15. GRAPH 7: TREND CONSISTENCY BY WELL
# ============================================================

trend_rows = []

for cell_line, well in series_keys:
    series = [
        r for r in growth_results
        if r["cell_line"] == cell_line and r["well"] == well
    ]

    series = sorted(series, key=lambda x: x["timepoint"])
    counts = np.array([r["cell_count"] for r in series])

    if len(counts) > 1:
        increasing_steps = int(np.sum(np.diff(counts) >= 0))
        possible_steps = len(counts) - 1
        monotonic_fraction = increasing_steps / possible_steps
    else:
        monotonic_fraction = np.nan

    trend_rows.append({
        "series": f"{cell_line}-{well}",
        "monotonic_fraction": monotonic_fraction
    })

trend_csv = FIGURE_FOLDER / "trend_consistency_summary.csv"

with open(trend_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["series", "monotonic_fraction"])
    writer.writeheader()
    writer.writerows(trend_rows)

plt.figure(figsize=(8, 5))
plt.bar(
    [r["series"] for r in trend_rows],
    [r["monotonic_fraction"] for r in trend_rows]
)
plt.ylim(0, 1.05)
plt.xlabel("Growth series")
plt.ylabel("Fraction of consecutive timepoints increasing")
plt.title("Internal QC: Trend Consistency by Well")
plt.xticks(rotation=25, ha="right")

save_plot("07_trend_consistency.png")

# ============================================================
# 16. PIPELINE EXAMPLE FIGURES
# ============================================================

def create_pipeline_figure(filename):
    image_path = IMAGE_FOLDER / filename

    if not image_path.exists():
        print("Example file not found, skipping:", filename)
        return

    result, images = count_cells(image_path)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(images["original"], cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(images["corrected"], cmap="gray")
    plt.title("Background Corrected")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(images["binary"], cmap="gray")
    plt.title("Binary Segmentation")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(images["annotated"], cv2.COLOR_BGR2RGB))
    plt.title(f"{result['status']}\nCount: {result['cell_count']}")
    plt.axis("off")

    safe_name = Path(filename).stem
    save_plot(f"pipeline_steps_{safe_name}.png")


example_candidates = [
    "UC-H7-t04.tif",
    "noGrowth1.tif",
    "AD-G5-t34.tif"
]

for example in example_candidates:
    create_pipeline_figure(example)

# ============================================================
# 17. SUMMARY TABLE FOR REPORT
# ============================================================

summary_filenames = [
    "AD-E1-t04.tif",
    "AD-E1-t34.tif",
    "AD-G5-t04.tif",
    "AD-G5-t34.tif",
    "UC-H7-t04.tif",
    "UC-H7-t34.tif",
    "noGrowth1.tif",
    "noGrowth2.tif"
]

summary_rows = [
    r for r in results
    if r["filename"] in summary_filenames
]

summary_csv = FIGURE_FOLDER / "summary_table_for_report.csv"

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "cell_line",
            "well",
            "timepoint",
            "cell_count",
            "status",
            "confidence_score",
            "total_area",
            "notes"
        ]
    )
    writer.writeheader()

    for row in summary_rows:
        writer.writerow({
            "filename": row["filename"],
            "cell_line": row["cell_line"],
            "well": row["well"],
            "timepoint": row["timepoint"],
            "cell_count": row["cell_count"],
            "status": row["status"],
            "confidence_score": row["confidence_score"],
            "total_area": row["total_area"],
            "notes": row["notes"]
        })

print("Summary table saved:", summary_csv)

# ============================================================
# 18. FINAL PRINT SUMMARY
# ============================================================

print("\n================ FINAL SUMMARY ================")
print("Total images analysed:", len(results))
print("Growth images:", len(growth_results))
print("No-growth controls:", len(control_results))

print("\nStatus counts:")
for status, count in status_counts.items():
    print(f"{status}: {count}")

print("\nInternal QC:")
print(f"Area-count correlation: {correlation:.3f}")

print("\nOutputs saved in:")
print(OUTPUT_FOLDER)