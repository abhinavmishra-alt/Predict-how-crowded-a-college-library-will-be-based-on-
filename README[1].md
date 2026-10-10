# QuietSpot ML 📚

**QuietSpot ML** is a small, professional machine-learning project that predicts
library crowd level from simple campus conditions.

The idea is intentionally practical: instead of asking *"Is the library crowded?"*,
a student can estimate the crowd level before walking there.

> **Educational note:** The included dataset is synthetic and designed for a
> reproducible ML demonstration. It is not real university usage data.

## Why this project is interesting

- Simple enough to understand in one sitting
- Uses a real ML workflow: data → preprocessing → training → evaluation → prediction
- Handles both numerical and categorical features
- Includes a small Streamlit interface
- Reproducible with a fixed random seed
- Easy to upload to GitHub and extend later

## ML task

This project treats occupancy prediction as a **regression problem**.

**Input features**
- Day of week
- Hour
- Exam-week flag
- Weather
- Semester week
- Number of nearby events

**Target**
- Library occupancy percentage (0–100)

The model uses a `RandomForestRegressor` because it is beginner-friendly,
works well with mixed patterns, and does not require feature scaling.

## Project structure

```text
quietspot-ml/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── data/
│   └── library_occupancy.csv
├── models/
│   └── README.md
└── assets/
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/quietspot-ml.git
cd quietspot-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This creates `models/quietspot_model.joblib`.

### 4. Start the web app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Model evaluation

`train_model.py` prints:

- MAE (Mean Absolute Error)
- R² score

The script uses an 80/20 train-test split and a fixed random state so the
results are reproducible.

## Future improvements

1. Replace the synthetic dataset with anonymized real library data.
2. Add a time-series model for hourly forecasting.
3. Show historical occupancy charts.
4. Add separate predictions for different library floors.
5. Deploy the Streamlit app publicly.

## Academic value

This mini-project demonstrates the complete beginner-to-intermediate ML
pipeline without unnecessary complexity:

**Problem → Dataset → Preprocessing → Model → Evaluation → Prediction → UI**

## License

MIT License.
