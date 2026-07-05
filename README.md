# Gym Performance Intelligence System

A beginner-friendly AI/ML portfolio project that analyzes a **synthetic** gym-performance dataset, generates exploratory charts, compares regression and classification models, calculates an explainable workout-readiness heuristic, and produces an interactive terminal report.

> **Synthetic-data disclaimer:** The included dataset is synthetic. It is not real human data and must not be described or interpreted as real participant data. The project is educational and does not provide medical, scientific, or causal conclusions.

## Problem Statement

Workout performance can vary alongside sleep, nutrition, hydration, stress, soreness, exercise selection, training history, and other factors. This project demonstrates how a complete beginner-level machine-learning workflow can inspect such relationships in synthetic data while avoiding obvious target leakage.

## Objectives

- Load and validate a structured CSV dataset.
- Clean data without modifying the raw CSV.
- Explore patterns with meaningful Matplotlib charts.
- Predict `actual_reps` with regression models.
- Predict `performance_class` with classification models.
- Compare models using appropriate metrics instead of assuming one algorithm is best.
- Calculate a transparent 0-100 experimental readiness score.
- Accept validated terminal input and generate a final performance report.

## Features

- Reliable project-relative file paths.
- Schema, missing-value, duplicate, category, range, and extreme-value checks.
- 12 EDA charts plus model-evaluation charts.
- Scikit-learn pipelines and `ColumnTransformer`.
- Train/test splitting before fitted preprocessing.
- One-hot encoding for `exercise`.
- Scaling for linear/logistic models where useful.
- Reproducible `random_state`.
- Interactive input validation.
- Automatic best-model selection from measured test results.

## Module A — Rep Predictor

Target: `actual_reps`

Models compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Metrics:

- MAE
- RMSE
- R² Score
- Training RMSE for a basic overfitting check

The best regression model is selected by lowest test RMSE. The program also saves a model-comparison chart, an actual-vs-predicted chart, and feature importance when the selected model supports it.

## Module B — Performance Classifier

Target: `performance_class`

Classes:

- Below Normal
- Normal
- Above Normal

Models compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Metrics:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score

The best classifier is selected by highest weighted F1-score. The program prints the class distribution and classification report, then saves a model-comparison chart and confusion matrix.

## Module C — Workout Readiness Score

`calculate_readiness_score()` is a transparent, editable rule-based heuristic from 0 to 100. It considers:

- Sleep duration
- Sleep quality
- Meal calories
- Carbohydrates
- Protein
- Meal timing
- Muscle soreness
- Recovery time for the same muscle
- Stress
- Hydration
- Caffeine

It returns:

- Overall score
- Individual factor scores
- Positive factors
- Limiting factors

This score is **experimental and educational**. It is not medically or scientifically validated.

## Dataset Description

File: `data/gym_performance_dataset_1500_rows.csv`

The included CSV contains 1,500 synthetic rows and 25 columns.

### Dataset Columns

- `date`
- `age`
- `height_cm`
- `body_weight_kg`
- `training_experience_months`
- `sleep_hours`
- `sleep_quality`
- `stress_level`
- `muscle_soreness`
- `days_since_last_training_same_muscle`
- `last_meal_calories`
- `last_meal_carbs_g`
- `last_meal_protein_g`
- `last_meal_fat_g`
- `minutes_since_last_meal`
- `water_before_workout_ml`
- `caffeine_mg`
- `exercise`
- `weight_lifted_kg`
- `set_number`
- `previous_session_reps`
- `recent_average_reps`
- `actual_reps`
- `performance_class`
- `readiness_score`

## Project Structure

```text
gym-performance-intelligence/
│
├── data/
│   └── gym_performance_dataset_1500_rows.csv
│
├── charts/
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

### 1. Open the project directory

```bash
cd gym-performance-intelligence
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

The program will:

1. Load the dataset.
2. Validate and clean a working copy.
3. Print the dataset overview.
4. Generate EDA charts.
5. Train and compare regression models.
6. Select the best regression model.
7. Train and compare classification models.
8. Select the best classification model.
9. Start the interactive terminal section.
10. Print the final Gym Performance Intelligence Report.

## Generated Charts

The program saves charts into `charts/`, including:

- Sleep hours vs actual reps
- Sleep quality vs performance
- Meal timing vs performance
- Carbohydrates vs actual reps
- Caffeine vs performance
- Soreness vs performance
- Stress vs performance
- Hydration vs readiness
- Set number vs performance
- Previous-session reps vs current reps
- Numerical correlation heatmap
- Exercise performance patterns
- Regression model comparison
- Actual vs predicted reps
- Regression feature importance when supported
- Classification model comparison
- Confusion matrix

## Data-Leakage Prevention

Leakage prevention is intentionally explicit.

For regression predicting `actual_reps`, these columns are excluded:

- `actual_reps` — the target itself
- `performance_class` — may be derived from actual performance
- `readiness_score` — synthetic score may encode information closely related to performance
- `date` — excluded from this beginner project rather than engineering time-derived features

For classification predicting `performance_class`, these columns are excluded:

- `performance_class` — the target itself
- `actual_reps` — may directly determine the class
- `readiness_score` — may encode outcome-related information
- `date`

`recent_average_reps` is retained because it represents historical baseline information available before the current result. The direct current outcome, `actual_reps`, is excluded. Preprocessing is fitted only inside Scikit-learn pipelines after the train/test split.

## Exact Model Scores

The program calculates and prints exact scores when it runs. This README does not hard-code scores because results should come from the executed dataset and environment.

## Limitations

- The dataset is synthetic, not real human data.
- Relationships in the data do not establish causation.
- The readiness score is a hand-built heuristic.
- The project uses a single train/test split rather than cross-validation.
- Hyperparameter tuning is intentionally limited to keep the project beginner-friendly.
- Predictions are educational and should not guide medical or safety-critical decisions.
- Exercise technique, injuries, equipment differences, and many real-world variables are not modeled.

## Future Improvements

- Add cross-validation.
- Add systematic hyperparameter tuning.
- Add confidence or prediction intervals.
- Engineer date-based features without leakage.
- Compare additional interpretable models.
- Add model calibration for classification.
- Test on independently collected, consented, properly governed data.
- Add automated tests while keeping the portfolio version easy to study.
