# 01_ImageProcessing

> A hands-on Digital Image Processing and Machine Vision portfolio project built with Python and OpenCV.

---

## Project Overview

This repository documents a step-by-step journey from the fundamentals of image processing to practical Machine Vision applications using Python and OpenCV.

The goal is not only to write example code, but also to build a structured understanding of techniques that can be applied in real industrial environments while developing a professional GitHub portfolio.

## Project Goals

- Learn image processing with Python
- Build practical experience with OpenCV
- Understand the fundamentals of Digital Image Processing
- Develop Machine Vision engineering skills
- Establish a foundation for AI Vision projects
- Build a well-organized GitHub portfolio

## Repository Structure

```text
01_ImageProcessing/
├── datasets/               # Datasets
├── docs/                   # Documentation
├── examples/               # OpenCV examples organized by chapter
│   ├── 01_Image_Basics/
│   ├── 02_Color_Space/
│   ├── 03_Histogram/
│   ├── 04_Filtering/
│   ├── 05_Threshold/
│   ├── 06_Morphology/
│   ├── 07_Edge_Detection/
│   └── 08_Contours/
├── images/                 # Sample images
├── notebooks/              # Jupyter notebooks
├── projects/               # Mini projects
├── scripts/                # Utility scripts
├── src/                    # Shared modules
└── tests/                  # Test code
```

## Current Examples

| Chapter | Topics |
|---|---|
| `01_Image_Basics` | Image loading, image information, and image saving |
| `02_Color_Space` | Color-space conversion and grayscale images |
| `03_Histogram` | Histogram calculation and visualization |
| `04_Filtering` | Average blur and Gaussian blur |
| `05_Threshold` | Binary thresholding |
| `06_Morphology` | Erosion, dilation, opening, closing, morphological gradient, Top-Hat, Black-Hat, and morphological kernel shape comparison |
| `07_Edge_Detection` | Edge detection techniques |
| `08_Contours` | Contour detection, area, perimeter, approximation, centroid, hierarchy, extreme points, and alignment analysis |

## Development Environment

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/milanpm/01_ImageProcessing.git
cd 01_ImageProcessing
pip install -r requirements.txt
```

## How to Run

Run an example from the repository root:

```bash
python examples/01_Image_Basics/src/01_image_load.py
```

Contour hierarchy example:

```bash
python examples/08_Contours/src/10_contour_hierarchy.py
```

Top-Hat transformation example:

```bash
python examples/06_Morphology/src/06_top_hat.py
```

Black-Hat transformation example:

```bash
python examples/06_Morphology/src/07_black_hat.py
```

Morphological kernel shape comparison example:

```bash
python examples/06_Morphology/src/08_compare_kernel_shapes.py
```

Centroid and extreme-point alignment example:

```bash
python examples/08_Contours/src/33_centroid_extreme_alignment.py
```

## Learning Roadmap

- [x] Image loading and information
- [x] Image saving
- [x] Color spaces
- [x] Histograms
- [x] Image filtering
- [x] Thresholding
- [x] Morphological operations
- [x] Edge detection
- [x] Contours and contour hierarchy
- [ ] Hough Transform
- [ ] Template Matching
- [ ] Feature Matching
- [ ] Camera Calibration
- [ ] Machine Vision
- [ ] Industrial Vision Project

## Requirements

The main dependencies are listed in `requirements.txt`:

```text
opencv-python
numpy
matplotlib
```

Install them with:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

## Author

**Alex**  
Machine Vision Engineer  
GitHub Portfolio Project
