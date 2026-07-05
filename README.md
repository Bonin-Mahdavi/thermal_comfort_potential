# Bonin–Mahdavi — Streetscape Thermal Comfort

This repository supports a comparative study of **visually assessed thermal comfort** in urban streetscapes across two case cities: **Detmold** (Germany) and **Turin** (Italy). Each city has ten survey points (five in LCZ-2, five in LCZ-5). Respondents rated how thermally comfortable they would expect to feel walking through each streetscape image on a hot summer day.

The project has two main parts:

1. **`semantic_segmentation/`** — derive streetscape composition from survey images.
2. **`data_analysis/`** — clean survey and attribute data, then answer the research questions in Jupyter notebooks.

```
bonin-mahdavi/
├── semantic_segmentation/   # Image segmentation → streetscape pixel shares
│   ├── images/              # Input streetscape photos (not tracked in git)
│   ├── results/             # Segmentation outputs (generated)
│   ├── pipeline.py
│   ├── segment_nvidia.py
│   └── segment_common.py
└── data_analysis/           # Survey + attribute analysis
    ├── Preprocessing.ipynb
    ├── RQ1.ipynb
    ├── RQ2.ipynb
    ├── config.py
    ├── utils.py
    ├── figures/             # Generated plots
    └── tables/              # Generated tables
```

---

## semantic_segmentation

Streetscape images are segmented with **NVIDIA SegFormer-B5** (`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`), fine-tuned on the Cityscapes dataset. For each image, the pipeline:

- assigns every pixel to one of **19 Cityscapes classes** (road, sidewalk, building, vegetation, sky, etc.);
- writes a colour-coded segmentation map with a class legend; and
- records the **pixel fraction** of each class (values in \[0, 1\]).

These pixel shares become **Layer 3a — SVI Pixel %** in the attribute table used downstream in `data_analysis`.

## data_analysis

This module analyses **questionnaire responses** (thermal comfort ratings and respondent profiles) together with **point-level attributes**: remote-sensing / GIS indicators (Layer 1) and streetscape pixel shares from semantic segmentation (Layer 3a). Data are loaded from Google Sheets, harmonised across languages (English, German, Italian), and screened **within each city** before modelling.

Run the notebooks in order:

| Notebook | Purpose |
|----------|---------|
| **`Preprocessing.ipynb`** | Variation and collinearity screening for attributes and questionnaire fields; builds the predictor sets used in RQ1 and RQ2. |
| **`RQ1.ipynb`** | *Visual assessment of thermal comfort across cities and social groups* — inter-rater agreement, city-level comfort profiles, and comparisons across respondent profiles. |
| **`RQ2.ipynb`** | *Physical and personal drivers of perceived thermal comfort* — compares nine regression learners per city, then uses **SHAP** on the best model to rank streetscape attributes and questionnaire variables. |

Figures and tables are saved under `figures/` and `tables/`.

### Data sources

- **Questionnaire** — Google Sheets (Detmold and Turin; English and local-language versions), merged into a single harmonised table.
- **Attributes** — Google Sheet with point identifiers, LCZ class, coordinates, RS/GIS layers, and SVI pixel percentages (from segmentation).



---

## How the two parts connect

```
Streetscape images          Survey responses
       │                           │
       ▼                           │
semantic_segmentation              │
  (SegFormer → pixel %)            │
       │                           │
       └──────────┬────────────────┘
                  ▼
           data_analysis
     (Preprocessing → RQ1 → RQ2)
```

Segmentation produces class-level pixel fractions (`nvidia-stats.xlsx`). Those values are incorporated into the attribute table as streetscape descriptors (e.g. `Vegetation (%)`, `Building (%)`, `Sky (%)`) and enter collinearity screening and the predictive models in RQ2 alongside RS/GIS predictors and questionnaire variables.

---

## Requirements

Each subfolder has its own `requirements.txt`:

- **`semantic_segmentation`** — PyTorch, Hugging Face Transformers, Pillow, openpyxl.
- **`data_analysis`** — pandas, scikit-learn, matplotlib, seaborn, SHAP, LightGBM, XGBoost, CatBoost, Jupyter.
