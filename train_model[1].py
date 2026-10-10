from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "library_occupancy.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "quietspot_model.joblib"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["occupancy_percent"])
y = df["occupancy_percent"]

categorical_features = ["day", "weather"]
numeric_features = [
    "hour",
    "exam_week",
    "semester_week",
    "nearby_events",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=180,
    random_state=42,
    max_depth=10,
    min_samples_leaf=3,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("QuietSpot ML - Model Training")
print("-" * 35)
print(f"MAE: {mae:.2f} percentage points")
print(f"R² : {r2:.3f}")
print(f"Saved model to: {MODEL_PATH}")
