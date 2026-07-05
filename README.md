# Gym Performance Intelligence Website

A complete end-to-end beginner AI/ML website built with FastAPI, HTML, CSS, JavaScript, Pandas, Matplotlib and Scikit-learn.

> The dataset is synthetic. The project is educational. The readiness score is an experimental heuristic and is not medically or scientifically validated.

## Structure

```text
gym-performance-website/
├── data/
│   └── gym_performance_dataset_1500_rows.csv
├── charts/
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html
├── app.py
├── ml_engine.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup on Windows

```bash
cd gym-performance-website
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place `gym_performance_dataset_1500_rows.csv` inside the `data` folder.

## Run

```bash
uvicorn app:app --reload
```

Open the local address shown in the terminal, normally `http://127.0.0.1:8000`.

The ML models train once when the server starts. The browser form then sends workout data to the FastAPI backend, which returns predicted reps, expected performance class and the experimental readiness report.

## Architecture

Browser form → FastAPI backend → existing Scikit-learn pipelines → prediction + readiness heuristic → rendered report.

## Important files

- `app.py`: web server, routes, form handling and predictions.
- `ml_engine.py`: your existing ML/data-analysis code.
- `templates/index.html`: website page and prediction form.
- `static/style.css`: responsive visual design.
- `static/script.js`: small browser interaction.
