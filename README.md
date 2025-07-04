# teams-stitching_analysis

## Prerequisites
* Python ≥ 3.8
* pip (Python package manager)

---

## Installation
```bash
# 1. Clone the repository (skip if you already have the folder)
# git clone <REPO_URL>
# cd teams-stitching_analysis

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage
```bash
python main.py
```

---

## Project Structure
```python
teams-stitching_analysis/
├─ assets/
│  ├─ test_image.png       # Default test image
│  └─ output/              # Result images produced at runtime
├─ image_processor.py      # Color-bar and edge detection functions
├─ marker_processor.py     # Marker detection functions
├─ utils.py                # Helper utilities (load / show / save images)
├─ main.py                 # Entry point
└─ requirements.txt        # Dependency list
```

---

## Custom Input Image
Place your own image in the `assets/` folder and adjust the path in `utils.py` → `load_image()` if necessary.

---

## Output Files
During processing the following files are saved to `assets/output/` (among others):
* `pattern_matching_result.png` – marker detection result
* `contours.png` – ROI contour
* `<color>_edge.png` – binary edge image for each color bar (red / green / blue / gray)
* `roi_edges.png` – ROI with all color bar edges drawn in different colors

---

