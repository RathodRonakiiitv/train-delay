# Indian Railways Train Delay Prediction

Kaggle Competition: Predict train delays (>15 minutes) for Indian Railways journeys.

## Project Structure

```
train-delay-prediction/
├── data/                   # Dataset files
│   ├── ir_train.csv       # 1.5M training records
│   ├── ir_test.csv        # 375K test records
│   ├── ir_sample_submission.csv
│   └── ir_data_dictionary.csv
├── notebooks/             # Jupyter notebooks
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── src/                   # Python modules
│   ├── data_utils.py     # Data loading utilities
│   ├── features.py       # Feature engineering
│   ├── models.py         # Model definitions
│   └── submission.py     # Submission generation
├── models/               # Saved model files
└── submissions/          # Kaggle submission files
```

## Competition Details

- **Target:** Predict probability of delay (>15 minutes)
- **Metric:** AUC-ROC
- **Training:** 1,500,000 records
- **Test:** 375,000 records
- **Features:** 42 (time, geography, route, weather, rolling stock, operations)

## Key Patterns to Exploit

1. **Train type priority:** Vande Bharat/Rajdhani have lower delay rates
2. **Zone congestion:** ER and NR are worst (~45-50% OTP)
3. **Monsoon effect:** July-August highest delays
4. **Fog cascade:** Dec-Feb morning delays in NR/NER/NCR
5. **Late incoming rake:** >80% delay probability if previous service delayed
6. **LHB coaches:** Better punctuality than ICF

## Quick Start

1. Download dataset from Kaggle to `data/` folder
2. Run `notebooks/01_eda.ipynb` to explore data
3. Run `notebooks/04_modeling.ipynb` to train models

## Data Dictionary

See `data/ir_data_dictionary.csv` for full feature descriptions.
