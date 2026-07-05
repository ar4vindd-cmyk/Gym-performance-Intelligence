from pathlib import Path
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "gym_performance_dataset_1500_rows.csv"
CHARTS_DIR = BASE_DIR / "charts"
RANDOM_STATE = 42

EXPECTED_COLUMNS = [
    "date", "age", "height_cm", "body_weight_kg", "training_experience_months",
    "sleep_hours", "sleep_quality", "stress_level", "muscle_soreness",
    "days_since_last_training_same_muscle", "last_meal_calories",
    "last_meal_carbs_g", "last_meal_protein_g", "last_meal_fat_g",
    "minutes_since_last_meal", "water_before_workout_ml", "caffeine_mg",
    "exercise", "weight_lifted_kg", "set_number", "previous_session_reps",
    "recent_average_reps", "actual_reps", "performance_class", "readiness_score"
]

REGRESSION_EXCLUDED = {"actual_reps", "performance_class", "readiness_score", "date"}
CLASSIFICATION_EXCLUDED = {
    "performance_class", "actual_reps", "readiness_score", "date"
}
VALID_PERFORMANCE_CLASSES = {"Below Normal", "Normal", "Above Normal"}

RANGES = {
    "age": (14, 100), "height_cm": (120, 230), "body_weight_kg": (30, 300),
    "training_experience_months": (0, 600), "sleep_hours": (0, 16),
    "sleep_quality": (1, 5), "stress_level": (1, 5), "muscle_soreness": (1, 5),
    "days_since_last_training_same_muscle": (0, 30),
    "last_meal_calories": (0, 3000), "last_meal_carbs_g": (0, 500),
    "last_meal_protein_g": (0, 250), "last_meal_fat_g": (0, 250),
    "minutes_since_last_meal": (0, 1440), "water_before_workout_ml": (0, 5000),
    "caffeine_mg": (0, 1000), "weight_lifted_kg": (0, 1000),
    "set_number": (1, 20), "previous_session_reps": (0, 100),
    "recent_average_reps": (0, 100), "actual_reps": (0, 100),
    "readiness_score": (0, 100)
}


# ============================================================
# 2. DATASET LOADING
# ============================================================

def load_data():
    """Load the synthetic gym-performance CSV using a reliable project-relative path."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n"
            "Place gym_performance_dataset_1500_rows.csv inside the data folder."
        )
    try:
        data = pd.read_csv(DATA_PATH)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The dataset file is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError("The dataset could not be parsed as a valid CSV file.") from error
    print(f"\nDataset loaded successfully from: {DATA_PATH}")
    return data


# ============================================================
# 3. DATA VALIDATION AND CLEANING
# ============================================================

def validate_and_clean_data(data):
    """Validate schema, types, categories and reasonable ranges without changing the raw CSV."""
    cleaned = data.copy()
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(cleaned.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned = cleaned[EXPECTED_COLUMNS]
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["exercise"] = cleaned["exercise"].astype("string").str.strip().str.title()
    cleaned["performance_class"] = (
        cleaned["performance_class"].astype("string").str.strip().str.title()
    )

    numeric_columns = [column for column in EXPECTED_COLUMNS
                       if column not in {"date", "exercise", "performance_class"}]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    print("\nMissing values before cleaning:")
    print(cleaned.isna().sum().to_string())

    duplicate_count = cleaned.duplicated().sum()
    print(f"\nDuplicate rows found: {duplicate_count}")
    if duplicate_count:
        cleaned = cleaned.drop_duplicates().copy()

    for column, (minimum, maximum) in RANGES.items():
        invalid_mask = cleaned[column].notna() & ~cleaned[column].between(minimum, maximum)
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            print(f"Setting {invalid_count} impossible value(s) in '{column}' to missing.")
            cleaned.loc[invalid_mask, column] = np.nan

    invalid_classes = set(cleaned["performance_class"].dropna().unique()) - VALID_PERFORMANCE_CLASSES
    if invalid_classes:
        print(f"Removing rows with inconsistent performance classes: {sorted(invalid_classes)}")
        cleaned = cleaned[~cleaned["performance_class"].isin(invalid_classes)].copy()

    cleaned = cleaned.dropna(subset=["exercise", "actual_reps", "performance_class"]).copy()
    if cleaned.empty:
        raise ValueError("No usable rows remain after validation.")

    # Extreme values are reported with the IQR rule but not blindly deleted.
    print("\nPotential extreme values (IQR rule, retained for analysis):")
    for column in ["sleep_hours", "last_meal_calories", "caffeine_mg",
                   "weight_lifted_kg", "actual_reps"]:
        series = cleaned[column].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        print(f"  {column}: {count}")

    return cleaned


def explore_data(data):
    """Print a useful overview of the cleaned dataset."""
    print("\n" + "=" * 72)
    print("DATASET OVERVIEW")
    print("=" * 72)
    print(f"Shape: {data.shape}")
    print("\nColumn names:")
    print(list(data.columns))
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))
    print("\nDescriptive statistics:")
    print(data.describe(include="all").transpose().to_string())
    print("\nMissing values after cleaning:")
    print(data.isna().sum().to_string())
    print("\nExercise values:")
    print(data["exercise"].value_counts().to_string())
    print("\nPerformance-class values:")
    print(data["performance_class"].value_counts().to_string())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

def perform_eda(data):
    """Create and save meaningful exploratory charts."""
    CHARTS_DIR.mkdir(exist_ok=True)
    for old_chart in CHARTS_DIR.glob("*.png"):
        old_chart.unlink()

    def save_chart(filename):
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(data["sleep_hours"], data["actual_reps"], alpha=0.35)
    plt.title("Sleep Duration vs Actual Repetitions")
    plt.xlabel("Sleep Hours")
    plt.ylabel("Actual Repetitions")
    save_chart("01_sleep_hours_vs_actual_reps.png")

    sleep_quality_means = data.groupby("sleep_quality")["actual_reps"].mean()
    plt.figure(figsize=(7, 5))
    plt.bar(sleep_quality_means.index.astype(str), sleep_quality_means.values)
    plt.title("Average Repetitions by Sleep Quality")
    plt.xlabel("Sleep Quality (1-5)")
    plt.ylabel("Average Actual Repetitions")
    save_chart("02_sleep_quality_vs_performance.png")

    bins = [0, 60, 120, 180, 240, np.inf]
    labels = ["<60", "60-120", "121-180", "181-240", "240+"]
    meal_groups = pd.cut(data["minutes_since_last_meal"], bins=bins, labels=labels)
    meal_means = data.groupby(meal_groups, observed=False)["actual_reps"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(meal_means.index.astype(str), meal_means.values)
    plt.title("Meal Timing and Average Performance")
    plt.xlabel("Minutes Since Last Meal")
    plt.ylabel("Average Actual Repetitions")
    save_chart("03_meal_timing_vs_performance.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(data["last_meal_carbs_g"], data["actual_reps"], alpha=0.35)
    plt.title("Pre-Workout Carbohydrates vs Actual Repetitions")
    plt.xlabel("Last Meal Carbohydrates (g)")
    plt.ylabel("Actual Repetitions")
    save_chart("04_carbs_vs_actual_reps.png")

    caffeine_means = data.groupby("caffeine_mg")["actual_reps"].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(caffeine_means.index, caffeine_means.values, marker="o")
    plt.title("Caffeine Intake and Average Repetitions")
    plt.xlabel("Caffeine (mg)")
    plt.ylabel("Average Actual Repetitions")
    save_chart("05_caffeine_vs_performance.png")

    soreness_groups = [data.loc[data["muscle_soreness"] == level, "actual_reps"].dropna()
                       for level in sorted(data["muscle_soreness"].dropna().unique())]
    plt.figure(figsize=(8, 5))
    plt.boxplot(soreness_groups, tick_labels=[str(x) for x in sorted(data["muscle_soreness"].dropna().unique())])
    plt.title("Muscle Soreness vs Performance")
    plt.xlabel("Muscle Soreness (1-5)")
    plt.ylabel("Actual Repetitions")
    save_chart("06_soreness_vs_performance.png")

    stress_means = data.groupby("stress_level")["actual_reps"].mean()
    plt.figure(figsize=(7, 5))
    plt.bar(stress_means.index.astype(str), stress_means.values)
    plt.title("Stress Level and Average Performance")
    plt.xlabel("Stress Level (1-5)")
    plt.ylabel("Average Actual Repetitions")
    save_chart("07_stress_vs_performance.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(data["water_before_workout_ml"], data["readiness_score"], alpha=0.35)
    plt.title("Hydration vs Dataset Readiness Score")
    plt.xlabel("Water Before Workout (ml)")
    plt.ylabel("Dataset Readiness Score")
    save_chart("08_hydration_vs_readiness.png")

    set_means = data.groupby("set_number")["actual_reps"].mean()
    plt.figure(figsize=(7, 5))
    plt.plot(set_means.index, set_means.values, marker="o")
    plt.title("Performance Across Set Numbers")
    plt.xlabel("Set Number")
    plt.ylabel("Average Actual Repetitions")
    save_chart("09_set_number_vs_performance.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(data["previous_session_reps"], data["actual_reps"], alpha=0.35)
    plt.title("Previous-Session Reps vs Current Reps")
    plt.xlabel("Previous-Session Repetitions")
    plt.ylabel("Actual Repetitions")
    save_chart("10_previous_vs_current_reps.png")

    numeric = data.select_dtypes(include=np.number)
    correlations = numeric.corr()
    plt.figure(figsize=(13, 10))
    image = plt.imshow(correlations, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(image, label="Correlation")
    plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(correlations.columns)), correlations.columns, fontsize=7)
    plt.title("Numerical Feature Correlation Heatmap")
    plt.xlabel("Numerical Features")
    plt.ylabel("Numerical Features")
    save_chart("11_correlation_heatmap.png")

    exercise_means = data.groupby("exercise")["actual_reps"].mean().sort_values()
    plt.figure(figsize=(9, 6))
    plt.barh(exercise_means.index, exercise_means.values)
    plt.title("Average Repetitions Across Exercises")
    plt.xlabel("Average Actual Repetitions")
    plt.ylabel("Exercise")
    save_chart("12_exercise_performance_patterns.png")

    print(f"\nEDA complete. Charts saved to: {CHARTS_DIR}")


# ============================================================
# 5. REGRESSION FEATURE PREPARATION
# ============================================================

def prepare_regression_data(data):
    """Prepare leakage-safe regression features and split before fitted preprocessing."""
    feature_columns = [c for c in data.columns if c not in REGRESSION_EXCLUDED]
    X = data[feature_columns].copy()
    y = data["actual_reps"].copy()
    categorical_features = ["exercise"]
    numerical_features = [c for c in feature_columns if c not in categorical_features]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE), numerical_features, categorical_features


def build_preprocessor(numerical_features, categorical_features, scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numerical_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])


# ============================================================
# 6. REGRESSION MODEL TRAINING AND EVALUATION
# ============================================================

def train_regression_models(split_data, numerical_features, categorical_features):
    """Train, compare and select regression models using test RMSE."""
    X_train, X_test, y_train, y_test = split_data
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }
    results, fitted = [], {}
    for name, model in models.items():
        preprocessor = build_preprocessor(
            numerical_features, categorical_features,
            scale_numeric=(name == "Linear Regression")
        )
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)
        results.append({
            "Model": name,
            "Train RMSE": mean_squared_error(y_train, train_predictions) ** 0.5,
            "Test MAE": mean_absolute_error(y_test, test_predictions),
            "Test RMSE": mean_squared_error(y_test, test_predictions) ** 0.5,
            "Test R2": r2_score(y_test, test_predictions)
        })
        fitted[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("Test RMSE")
    print("\nREGRESSION MODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nSelected regression model: {best_name}")

    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["Test RMSE"])
    plt.title("Regression Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test RMSE (Lower Is Better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "13_regression_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_predictions = best_model.predict(X_test)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_predictions, alpha=0.5)
    limits = [min(y_test.min(), best_predictions.min()), max(y_test.max(), best_predictions.max())]
    plt.plot(limits, limits, linestyle="--")
    plt.title(f"Actual vs Predicted Reps — {best_name}")
    plt.xlabel("Actual Repetitions")
    plt.ylabel("Predicted Repetitions")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "14_actual_vs_predicted_reps.png", dpi=150, bbox_inches="tight")
    plt.close()

    model_step = best_model.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
        importances = pd.Series(model_step.feature_importances_, index=feature_names).nlargest(15).sort_values()
        plt.figure(figsize=(9, 7))
        plt.barh(importances.index, importances.values)
        plt.title(f"Top Regression Feature Importances — {best_name}")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "15_regression_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    return best_model, [c for c in X_train.columns]


# ============================================================
# 7. CLASSIFICATION FEATURE PREPARATION
# ============================================================

def prepare_classification_data(data):
    """Prepare classification data while excluding direct target-derived leakage."""
    feature_columns = [c for c in data.columns if c not in CLASSIFICATION_EXCLUDED]
    X = data[feature_columns].copy()
    y = data["performance_class"].copy()
    categorical_features = ["exercise"]
    numerical_features = [c for c in feature_columns if c not in categorical_features]
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    ), numerical_features, categorical_features


# ============================================================
# 8. CLASSIFICATION MODEL TRAINING AND EVALUATION
# ============================================================

def train_classification_models(split_data, numerical_features, categorical_features):
    """Train, compare and select classifiers using weighted F1-score."""
    X_train, X_test, y_train, y_test = split_data
    print("\nCLASS DISTRIBUTION")
    print(y_train.value_counts().to_string())

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=250, max_depth=14, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }
    results, fitted = [], {}
    for name, model in models.items():
        preprocessor = build_preprocessor(
            numerical_features, categorical_features,
            scale_numeric=(name == "Logistic Regression")
        )
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Weighted Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Weighted Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "Weighted F1": f1_score(y_test, predictions, average="weighted", zero_division=0)
        })
        fitted[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("Weighted F1", ascending=False)
    print("\nCLASSIFICATION MODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)
    print(f"\nSelected classification model: {best_name}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["Weighted F1"])
    plt.title("Classification Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Weighted F1-Score (Higher Is Better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "16_classification_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    labels = ["Below Normal", "Normal", "Above Normal"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(7, 6))
    image = plt.imshow(matrix, cmap="Blues")
    plt.colorbar(image)
    plt.xticks(range(len(labels)), labels, rotation=25)
    plt.yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(column, row, matrix[row, column], ha="center", va="center")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "17_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    return best_model, [c for c in X_train.columns]


# ============================================================
# 9. READINESS SCORE CALCULATION
# ============================================================

def calculate_readiness_score(values):
    """Return an experimental, transparent 0-100 workout-readiness heuristic."""
    factor_scores = {}

    sleep_hours = values["sleep_hours"]
    factor_scores["Sleep duration"] = max(0, 100 - abs(sleep_hours - 8) * 22)
    factor_scores["Sleep quality"] = values["sleep_quality"] * 20
    factor_scores["Meal calories"] = max(0, 100 - abs(values["last_meal_calories"] - 650) / 6.5)
    factor_scores["Carbohydrates"] = max(0, 100 - abs(values["last_meal_carbs_g"] - 90) * 1.1)
    factor_scores["Protein"] = max(0, 100 - abs(values["last_meal_protein_g"] - 35) * 2.2)

    minutes = values["minutes_since_last_meal"]
    if 60 <= minutes <= 180:
        factor_scores["Meal timing"] = 100
    elif 30 <= minutes < 60 or 180 < minutes <= 240:
        factor_scores["Meal timing"] = 70
    else:
        factor_scores["Meal timing"] = 35

    factor_scores["Muscle soreness"] = max(0, 120 - values["muscle_soreness"] * 20)
    recovery_days = values["days_since_last_training_same_muscle"]
    factor_scores["Muscle recovery time"] = min(100, recovery_days * 30)
    factor_scores["Stress"] = max(0, 120 - values["stress_level"] * 20)
    factor_scores["Hydration"] = min(100, values["water_before_workout_ml"] / 8)

    caffeine = values["caffeine_mg"]
    if caffeine == 0:
        factor_scores["Caffeine"] = 70
    elif 50 <= caffeine <= 250:
        factor_scores["Caffeine"] = 90
    elif caffeine <= 400:
        factor_scores["Caffeine"] = 60
    else:
        factor_scores["Caffeine"] = 30

    weights = {
        "Sleep duration": 0.14, "Sleep quality": 0.10, "Meal calories": 0.07,
        "Carbohydrates": 0.10, "Protein": 0.06, "Meal timing": 0.10,
        "Muscle soreness": 0.12, "Muscle recovery time": 0.08,
        "Stress": 0.10, "Hydration": 0.09, "Caffeine": 0.04
    }
    overall = round(sum(factor_scores[name] * weights[name] for name in weights))
    overall = int(np.clip(overall, 0, 100))

    positives = [name for name, score in factor_scores.items() if score >= 80]
    limiting = [name for name, score in factor_scores.items() if score < 55]
    if not positives:
        positives = ["No factor reached the strong-positive threshold"]
    if not limiting:
        limiting = ["No major limiting factor detected by the heuristic"]

    return overall, {k: round(v, 1) for k, v in factor_scores.items()}, positives, limiting


# ============================================================
# 10. USER INPUT AND VALIDATION
# ============================================================

def get_user_input(data, required_features):
    """Collect and validate every feature needed by the trained models and heuristic."""
    available_exercises = sorted(data["exercise"].dropna().unique().tolist())
    print("\n" + "=" * 72)
    print("INTERACTIVE PREDICTION")
    print("=" * 72)
    print("Available exercises:", ", ".join(available_exercises))

    def ask_number(prompt, minimum, maximum, integer=False):
        while True:
            try:
                value = float(input(prompt).strip())
                if not minimum <= value <= maximum:
                    print(f"Enter a value from {minimum} to {maximum}.")
                    continue
                return int(value) if integer else value
            except ValueError:
                print("Enter a valid number.")

    while True:
        exercise_value = input("Exercise: ").strip().title()
        if exercise_value in available_exercises:
            break
        print("Invalid exercise. Choose one of:", ", ".join(available_exercises))

    values = {
        "age": ask_number("Age: ", 14, 100, True),
        "height_cm": ask_number("Height (cm): ", 120, 230),
        "body_weight_kg": ask_number("Body weight (kg): ", 30, 300),
        "training_experience_months": ask_number("Training experience (months): ", 0, 600, True),
        "sleep_hours": ask_number("Sleep hours: ", 0, 16),
        "sleep_quality": ask_number("Sleep quality (1-5): ", 1, 5, True),
        "stress_level": ask_number("Stress level (1-5): ", 1, 5, True),
        "muscle_soreness": ask_number("Muscle soreness (1-5): ", 1, 5, True),
        "days_since_last_training_same_muscle": ask_number("Days since training same muscle: ", 0, 30, True),
        "last_meal_calories": ask_number("Last meal calories: ", 0, 3000),
        "last_meal_carbs_g": ask_number("Last meal carbohydrates (g): ", 0, 500),
        "last_meal_protein_g": ask_number("Last meal protein (g): ", 0, 250),
        "last_meal_fat_g": ask_number("Last meal fat (g): ", 0, 250),
        "minutes_since_last_meal": ask_number("Minutes since last meal: ", 0, 1440, True),
        "water_before_workout_ml": ask_number("Water before workout (ml): ", 0, 5000),
        "caffeine_mg": ask_number("Caffeine (mg): ", 0, 1000),
        "exercise": exercise_value,
        "weight_lifted_kg": ask_number("Training load / weight lifted (kg): ", 0, 1000),
        "set_number": ask_number("Set number: ", 1, 20, True),
        "previous_session_reps": ask_number("Previous-session reps: ", 0, 100, True),
        "recent_average_reps": ask_number("Recent average reps: ", 0, 100)
    }
    return {feature: values[feature] for feature in required_features}


# ============================================================
# 11. FINAL PREDICTION REPORT
# ============================================================

def generate_performance_report(user_values, regression_model, classification_model,
                                regression_features, classification_features):
    """Generate model predictions, heuristic readiness and the final terminal report."""
    regression_input = pd.DataFrame([{f: user_values[f] for f in regression_features}])
    classification_input = pd.DataFrame([{f: user_values[f] for f in classification_features}])

    predicted_reps = max(0, float(regression_model.predict(regression_input)[0]))
    predicted_class = classification_model.predict(classification_input)[0]
    readiness, factor_scores, positives, limiting = calculate_readiness_score(user_values)

    print("\n" + "=" * 48)
    print("GYM PERFORMANCE INTELLIGENCE REPORT")
    print("=" * 48)
    print(f"\nExercise: {user_values['exercise']}")
    print(f"Training Load: {user_values['weight_lifted_kg']:.1f} kg")
    print(f"\nPredicted Reps: {round(predicted_reps)}")
    print(f"Expected Performance: {predicted_class}")
    print(f"Workout Readiness: {readiness}/100")
    print("\nIndividual Readiness Factors:")
    for factor, score in factor_scores.items():
        print(f"  - {factor}: {score}/100")
    print("\nPositive Factors:")
    for factor in positives:
        print(f"  - {factor}")
    print("\nLimiting Factors:")
    for factor in limiting:
        print(f"  - {factor}")
    print("\nNote: The readiness score is an experimental heuristic, not a medical")
    print("or scientifically validated assessment.")
    print("=" * 48)


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():
    """Run the complete Gym Performance Intelligence System."""
    try:
        data = load_data()
        data = validate_and_clean_data(data)
        explore_data(data)
        perform_eda(data)

        regression_split, regression_numeric, regression_categorical = prepare_regression_data(data)
        regression_model, regression_features = train_regression_models(
            regression_split, regression_numeric, regression_categorical
        )

        classification_split, classification_numeric, classification_categorical = prepare_classification_data(data)
        classification_model, classification_features = train_classification_models(
            classification_split, classification_numeric, classification_categorical
        )

        all_required_features = list(dict.fromkeys(regression_features + classification_features))
        user_values = get_user_input(data, all_required_features)
        generate_performance_report(
            user_values, regression_model, classification_model,
            regression_features, classification_features
        )
        print("\nProgram finished successfully.")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nProgram stopped by the user. Exiting cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
