# Music Genre Classification

A machine learning project that classifies music into 10 genres by extracting audio features and comparing three different classifiers: **K-Nearest Neighbors (KNN)**, **Multi-Layer Perceptron (MLP)**, and **Support Vector Machine (SVM)**.

Models are trained on the GTZAN dataset and evaluated on FMA songs across multiple segment durations (3, 6, and 10 seconds) to analyze how audio context length affects classification accuracy.

---

## File Structure

```
Project/
├── src/
│   └── extract_features.py        # Core audio feature extraction utility
│
├── notebooks/
│   ├── gtzan_dataset.ipynb        # Dataset exploration and visualization
│   ├── knn_model.ipynb            # KNN training with hyperparameter tuning
│   ├── mlp_model.ipynb            # MLP (neural network) training
│   ├── svm_model.ipynb            # SVM training
│   └── compare_models.ipynb       # Cross-model and cross-duration evaluation
│
├── data/
│   ├── features_3_sec.csv         # GTZAN features extracted at 3-second segments
│   ├── features_30_sec.csv        # GTZAN features extracted at 30-second segments
│   ├── genres_original/           # GTZAN dataset — 1000 WAV files, 100 per genre
│   ├── fma_songs/                 # FMA dataset — 200 MP3 files, 20 per genre
│   ├── images_original/           # Spectrograms per genre (for visualization)
│   ├── custom_songs/              # Arbitrary songs for ad-hoc testing
│   └── results/                   # Generated charts and evaluation CSVs
│
└── models/                        # Saved model pipelines (KNN_3_sec.joblib, etc.)
```

---

## Datasets

| Dataset | Files | Purpose |
|---------|-------|---------|
| **GTZAN** | 1000 WAV (30 sec each) | Primary training data |
| **FMA** | 200 MP3 | External evaluation / generalization testing |
| **Custom Songs** | 3 files | Ad-hoc single-song testing |

Both datasets cover 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
<br>
GTZAN dataset can be downloaded from [here](https://www.tensorflow.org/datasets/catalog/gtzan).
FMA songs were obtained from [here](https://freemusicarchive.org/).

---

## Feature Extraction

`src/extract_features.py` provides a single function:

```python
extract_features(file_path, label, sr=22050, segment_duration=3)
```

It loads an audio file, splits it into fixed-length segments, and extracts **58 features** per segment:

| Feature Group | Features |
|---------------|----------|
| Chroma STFT | mean, var |
| RMS Energy | mean, var |
| Spectral Centroid | mean, var |
| Spectral Bandwidth | mean, var |
| Spectral Rolloff | mean, var |
| Zero Crossing Rate | mean, var |
| Harmonic component | mean, var |
| Percussive component | mean, var |
| Tempo | 1 value |
| MFCCs (20 coefficients) | mean + var each = 40 |

Returns a pandas DataFrame with one row per segment.

---

## Workflow

```
Raw Audio
    └─ extract_features.py
         └─ features_3_sec.csv / features_30_sec.csv
              ├─ knn_model.ipynb   → KNN_*.joblib
              ├─ mlp_model.ipynb   → MLP_*.joblib
              └─ svm_model.ipynb   → SVM_*.joblib
                   └─ compare_models.ipynb
                        └─ data/results/  (charts + CSVs)
```

1. **Feature Extraction** — Run `extract_features()` on the GTZAN audio files to produce the feature CSVs.
2. **Model Training** — Each model notebook trains a classifier on `features_3_sec.csv`, tunes hyperparameters, and saves a `.joblib` file.
3. **Evaluation** — `compare_models.ipynb` loads all saved models, extracts features from the FMA dataset at 3, 6, and 10-second segment durations, and evaluates accuracy per model, per genre, and per duration.

---

## Notebooks

### `gtzan_dataset.ipynb`
Exploratory data analysis: feature statistics, waveform plots, spectrograms, and chroma STFT visualizations per genre.

### `knn_model.ipynb`
Trains a KNN classifier. Uses `RandomizedSearchCV` to tune `n_neighbors`, `weights`, and the Minkowski distance parameter `p`. Saves the final pipeline as `KNN_3_sec.joblib`.

### `mlp_model.ipynb`
Trains a Keras neural network. Applies `StandardScaler` and label encoding, then fits an MLP. Saves the pipeline as `MLP_3_sec.joblib`.

### `svm_model.ipynb`
Trains an SVM classifier using a modular `train_model()` function. Saves the pipeline as `SVM_3_sec.joblib`.

### `compare_models.ipynb`
The main evaluation notebook. For each segment duration (3, 6, 10 sec):
- Extracts features from the 200 FMA songs
- Runs predictions through KNN, MLP, and SVM
- Computes overall accuracy, per-genre accuracy, and confusion matrices
- Saves results to `data/results/`

**Output files in `data/results/`:**
- `accuracy_overall_*sec.png` — Bar chart of overall accuracy
- `accuracy_per_genre_*sec.png` — Per-genre accuracy breakdown
- `confusion_matrices_*sec.png` — 3×3 grid of confusion matrices
- `accuracy_trend_all_segments.png` — Accuracy vs. segment duration
- `accuracy_heatmap_all_segments.png` — Heatmap of all results
- `fma_model_comparison_*sec.csv` — Raw prediction data

---

## Installation

```bash
pip install -e .
```

Dependencies (from `pyproject.toml`):

- `librosa` — audio loading and feature extraction
- `scikit-learn` — KNN, SVM, preprocessing, evaluation
- `keras` / `torch` — MLP neural network
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualization
- `joblib` — model serialization

**Python >= 3.11**

---