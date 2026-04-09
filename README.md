# Project 2: Large-Scale Unsupervised Structure Discovery on the UCI HIGGS Dataset

**Course:** CS 4/5630 - Python for Computational and Data Sciences
**Instructor:** Dr. Arijit Khan
**Group 1:** Quang Minh Nguyen, Rachel Stevenson, Jack Handley, Isaac Avila, Vaughn Gugger, Chaz Wilms

---

## Project Overview

This project implements an unsupervised machine learning pipeline to explore the underlying structure of the UCI HIGGS dataset. By using k-Means clustering and Principal Component Analysis (PCA), the pipeline analyzes how dimensionality reduction affects clustering quality, computational runtime, and variance retention. Given the scale of the dataset (11 million rows), the project demonstrates strategies like subsampling and scaling to maintain analytical integrity at scale.

---

## Datasets

**You must obtain the raw data files before running the pipeline.** The data is not stored in this repository due to file size. Download it from the shared OneDrive folder:

**[Download Raw Data (OneDrive)](https://falconbgsu-my.sharepoint.com/personal/cwilms_bgsu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcwilms%5Fbgsu%5Fedu%2FDocuments%2FCS4630%20Group%201%20%2D%20Data%2FProject%202&viewid=94f93628%2Da265%2D4f13%2Da339%2Dab773cd74794)**

Once downloaded, place the files in the repository as follows:

```
data/
└──  HIGGS.csv.gz  <- Raw compressed dataset (11M rows, 28 features)
```

---

## Repository Structure

```
CS4630Group1Project2/
├── data/
│   └── HIGGS.csv.gz            # Raw input file (not committed)
├── labels/                     # Per-run cluster labels (generated in Step 2/4)
├── outputs/                    # All intermediate CSVs and visualization plots
├── src/
│   ├── clustering.py           # Core k-Means logic and metric calculations
│   ├── pca_module.py           # PCA transformation functions
│   ├── preprocessing.py        # Data loading and scaling functions
│   ├── step1_preprocess.py     # Phase 1: Subsampling and Standard Scaling
│   ├── step2_cluster_raw.py    # Phase 2: k-Means on 28D feature space
│   ├── step3_pca.py            # Phase 3: Dimensionality reduction (2, 5, 10 components)
│   ├── step4_cluster_pca.py    # Phase 4: k-Means on PCA-reduced data
│   ├── step5_visualization.py  # Phase 5: Analytical plotting and comparisons
│   └── step6_combine.py        # Phase 6: Result aggregation and analysis summary
├── .gitignore
└── README.md
```

---

## Setup

### 1. Python Version

Python 3.9 or higher is recommended.

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

---

## How to Run the Pipeline

Run all scripts from the **project root directory** (CS4630Group1Project2/) unless noted otherwise.

---

### Phase 1 - Data Preprocessing

**Subsample and scale the dataset:** 
```bash
python src/step1_preprocess.py
```
- Input: `data/HIGGS.csv.gz`
- Output: `outputs/X_scaled.csv`, `outputs/y.csv`
- Operations: Loads 200,000 rows, separates labels, and applies `StandardScalar`.

---

### Phase 2 - Raw Feature Clustering
**Run k-Means on the full 28-dimensional space:**
```bash
python src/step2_cluster_raw.py
```
- Input: `outputs/X_scaled.csv`
- Output: `outputs/clustering_raw_results.csv`, `figures/elbow_raw.png`
- Operations: Iterates k from 2 to 9; calculates Silhouette and Davies-Bouldin scores.

---

### Phase 3 - Dimensionality Reduction
**Apply PCA to reduce feature space:**
```bash
python src/step3_pca.py
```
- Input: `outputs/X_scaled.csv`
- Output: `outputs/X_pca_2.csv`, `outputs/X_pca_5.csv`, `outputs/X_pca_10.csv`
- Metrics: Prints total variance retained for each component count.

---

### Phase 4 - PCA Feature Clustering
**Run k-Means on reduced datasets:**
```bash
python src/step4_cluster_pca.py
```
- Input: All PCA-reduced CSVs in `outputs/`
- Output: `outputs/clustering_pca_results.csv`
- Operations: Repeats clustering k = 2 to 9 for each PCA variation

---

### Phase 5 - Visualization & Analysis
**Generate comparison plots:**
```bash
python src/step5_visualization.py
```
- Outputs: 
    - `pca_2d_clusters_advanced.png` (Visual 2D cluster separation)
    - `elbow_comparison.png` (Inertia comparison)
    - `silhouette_comparison.png` & `db_index_comparison.png` (Quality metrics)
    - `variance_explained.png` (PCA cumulative variance)

---

## Output Files Summary

| File | Description |
|---|---|
| `outputs/X_scaled.csv` | Scaled features for the 200k subsample |
| `outputs/clustering_raw_results.csv` | Performance metrics for 28D clustering | 
| `outputs/clustering_pca_results.csv` | Performance metrics for PCA-reduced clustering |
| `outputs/final_comparison.csv` | Consolidated metrics for all project runs |
| `figures/*.png` | Visualizations for report inclusion | 

---

## Key Findings

- **Clustering Quality:** Contrast between Silhouette scores in 28D vs. PCA-reduced space.
- **Dimensionality:** Number of components required to reach 80% and 90% variance thresholds.
- **Scalability:** Computational runtime (seconds) comparison between raw and reduced feature sets.