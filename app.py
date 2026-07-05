import io
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import ml_engine

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app_state = {
    "data": None,
    "regression_model": None,
    "classification_model": None,
    "regression_features": None,
    "classification_features": None,
    "exercises": [],
    "startup_error": None,
}


def train_models():
    with redirect_stdout(io.StringIO()):
        """Load the synthetic dataset and train both ML systems once at startup."""
        data = ml_engine.load_data()
        data = ml_engine.validate_and_clean_data(data)

        regression_split, regression_numeric, regression_categorical = (
            ml_engine.prepare_regression_data(data)
        )
        regression_model, regression_features = ml_engine.train_regression_models(
            regression_split, regression_numeric, regression_categorical
        )

        classification_split, classification_numeric, classification_categorical = (
            ml_engine.prepare_classification_data(data)
        )
        classification_model, classification_features = (
            ml_engine.train_classification_models(
                classification_split,
                classification_numeric,
                classification_categorical,
            )
        )

        app_state.update(
            {
                "data": data,
                "regression_model": regression_model,
                "classification_model": classification_model,
                "regression_features": regression_features,
                "classification_features": classification_features,
                "exercises": sorted(data["exercise"].dropna().unique().tolist()),
                "startup_error": None,
            }
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        train_models()
    except Exception as error:
        app_state["startup_error"] = str(error)
    yield


app = FastAPI(
    title="Gym Performance Intelligence System",
    description="Educational ML website using a synthetic dataset.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "exercises": app_state["exercises"],
            "startup_error": app_state["startup_error"],
            "result": None,
            "form": {},
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    height_cm: float = Form(...),
    body_weight_kg: float = Form(...),
    training_experience_months: int = Form(...),
    sleep_hours: float = Form(...),
    sleep_quality: int = Form(...),
    stress_level: int = Form(...),
    muscle_soreness: int = Form(...),
    days_since_last_training_same_muscle: int = Form(...),
    last_meal_calories: float = Form(...),
    last_meal_carbs_g: float = Form(...),
    last_meal_protein_g: float = Form(...),
    last_meal_fat_g: float = Form(...),
    minutes_since_last_meal: int = Form(...),
    water_before_workout_ml: float = Form(...),
    caffeine_mg: float = Form(...),
    exercise: str = Form(...),
    weight_lifted_kg: float = Form(...),
    set_number: int = Form(...),
    previous_session_reps: int = Form(...),
    recent_average_reps: float = Form(...),
):
    if app_state["startup_error"]:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "exercises": app_state["exercises"],
                "startup_error": app_state["startup_error"],
                "result": None,
                "form": {},
            },
            status_code=500,
        )

    values = {
        "age": age,
        "height_cm": height_cm,
        "body_weight_kg": body_weight_kg,
        "training_experience_months": training_experience_months,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "stress_level": stress_level,
        "muscle_soreness": muscle_soreness,
        "days_since_last_training_same_muscle": days_since_last_training_same_muscle,
        "last_meal_calories": last_meal_calories,
        "last_meal_carbs_g": last_meal_carbs_g,
        "last_meal_protein_g": last_meal_protein_g,
        "last_meal_fat_g": last_meal_fat_g,
        "minutes_since_last_meal": minutes_since_last_meal,
        "water_before_workout_ml": water_before_workout_ml,
        "caffeine_mg": caffeine_mg,
        "exercise": exercise,
        "weight_lifted_kg": weight_lifted_kg,
        "set_number": set_number,
        "previous_session_reps": previous_session_reps,
        "recent_average_reps": recent_average_reps,
    }

    if exercise not in app_state["exercises"]:
        error = "Choose an exercise from the available dataset values."
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "exercises": app_state["exercises"],
                "startup_error": error,
                "result": None,
                "form": values,
            },
            status_code=400,
        )

    regression_input = pd.DataFrame(
        [{feature: values[feature] for feature in app_state["regression_features"]}]
    )
    classification_input = pd.DataFrame(
        [{feature: values[feature] for feature in app_state["classification_features"]}]
    )

    predicted_reps = max(
        0.0, float(app_state["regression_model"].predict(regression_input)[0])
    )
    predicted_class = app_state["classification_model"].predict(
        classification_input
    )[0]
    readiness, factor_scores, positives, limiting = (
        ml_engine.calculate_readiness_score(values)
    )

    result = {
        "predicted_reps": round(predicted_reps),
        "predicted_class": predicted_class,
        "readiness": readiness,
        "factor_scores": factor_scores,
        "positives": positives,
        "limiting": limiting,
        "exercise": exercise,
        "weight_lifted_kg": weight_lifted_kg,
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "exercises": app_state["exercises"],
            "startup_error": None,
            "result": result,
            "form": values,
        },
    )
