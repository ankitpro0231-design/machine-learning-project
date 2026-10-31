"""
Train & evaluate house price models on the multi-city India housing dataset.
Run:  python train.py
Outputs: models/best_model.joblib, models/metrics.json, models/feature_importance.png,
         models/actual_vs_pred.png, models/meta.json (used by the Streamlit app)
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "indian_housing_synthetic_1k.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "price_lakhs"
FURNISHING_ORDER = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]


# ---------------------------------------------------------------- features
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived features. Never uses the target -> no leakage."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["floor_ratio"] = X["floor"] / X["total_floors"].clip(lower=1)
        X["is_top_floor"] = (X["floor"] >= X["total_floors"]).astype(int)
        X["is_ground_floor"] = (X["floor"] == 0).astype(int)
        X["area_per_bhk"] = X["area"] / X["BHK"].clip(lower=1)
        X["log_area"] = np.log1p(X["area"])
        X["is_new"] = (X["age"] <= 2).astype(int)
        X["near_metro"] = (X["distance_to_metro_km"] <= 1).astype(int)
        X["city_locality"] = X["city"] + "_" + X["locality"]
        return X


NUM_COLS = ["BHK", "area", "floor", "total_floors", "age", "latitude", "longitude",
            "distance_to_cbd_km", "distance_to_metro_km", "floor_ratio", "is_top_floor",
            "is_ground_floor", "area_per_bhk", "log_area", "is_new", "near_metro"]
CAT_COLS = ["city", "locality", "city_locality"]
ORD_COLS = ["furnishing"]


def make_preprocessor(scale=False):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUM_COLS),
        ("ord", OrdinalEncoder(categories=[FURNISHING_ORDER]), ORD_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])


def make_pipeline(model, scale=False):
    # log-transform the target: prices are right-skewed (11 to 1127 lakhs)
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("fe", FeatureEngineer()),
            ("prep", make_preprocessor(scale)),
            ("model", model),
        ]),
        func=np.log1p, inverse_func=np.expm1,
    )


def evaluate(y_true, y_pred):
    return {
        "MAE_lakhs": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_lakhs": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def main():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # basic sanity cleaning
    df = df[(df["floor"] <= df["total_floors"]) & (df[TARGET] > 0) & (df["area"] > 0)]

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    # stratify by city so every split has every city
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=df["city"])

    # ------------------------------------------------ baseline model comparison
    candidates = {
        "Ridge": make_pipeline(Ridge(alpha=1.0), scale=True),
        "RandomForest": make_pipeline(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "GradientBoosting": make_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
        "HistGradientBoosting": make_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    print("\n5-fold CV (R2 on train set):")
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_results[name] = {"cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std())}
        print(f"  {name:22s} {scores.mean():.4f} +/- {scores.std():.4f}")

    # ------------------------------------------------ tune Ridge and GradientBoosting
    # (dataset is synthetic & roughly log-linear, so a tuned linear model can win -
    #  we let cross-validation decide instead of assuming trees are better)
    print("\nTuning models ...")
    ridge_search = RandomizedSearchCV(
        make_pipeline(Ridge(), scale=True),
        param_distributions={"regressor__model__alpha": np.logspace(-3, 2, 30)},
        n_iter=20, cv=cv, scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE, n_jobs=-1)
    ridge_search.fit(X_train, y_train)

    gb_search = RandomizedSearchCV(
        make_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
        param_distributions={
            "regressor__model__n_estimators": [200, 300, 500, 800],
            "regressor__model__learning_rate": [0.02, 0.05, 0.08, 0.1],
            "regressor__model__max_depth": [2, 3, 4, 5],
            "regressor__model__min_samples_leaf": [1, 3, 5, 10],
            "regressor__model__subsample": [0.7, 0.85, 1.0],
        },
        n_iter=25, cv=cv, scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE, n_jobs=-1)
    gb_search.fit(X_train, y_train)

    print(f"Ridge tuned CV MAE: {-ridge_search.best_score_:.3f}  | GB tuned CV MAE: {-gb_search.best_score_:.3f}")
    tuned = {"Ridge (tuned)": ridge_search, "GradientBoosting (tuned)": gb_search}
    best_name = max(tuned, key=lambda k: tuned[k].best_score_)
    search = tuned[best_name]
    best = search.best_estimator_
    print("Selected model:", best_name, search.best_params_)

    # ------------------------------------------------ final evaluation on held-out test
    results = {"cv": cv_results, "test": {}}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        results["test"][name] = evaluate(y_test, pipe.predict(X_test))
    for name, srch in tuned.items():
        results["test"][name] = evaluate(y_test, srch.best_estimator_.predict(X_test))
    pred = best.predict(X_test)
    results["selected_model"] = best_name
    results["best_params"] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}

    print("\nTest-set results:")
    print(pd.DataFrame(results["test"]).T.round(3))

    # ------------------------------------------------ plots
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.6)
    lim = [0, max(y_test.max(), pred.max())]
    ax.plot(lim, lim, "r--")
    ax.set_xlabel("Actual price (lakhs)"); ax.set_ylabel("Predicted price (lakhs)")
    ax.set_title("Actual vs Predicted (test set)")
    fig.tight_layout(); fig.savefig(MODEL_DIR / "actual_vs_pred.png", dpi=150); plt.close(fig)

    # permutation importance on ORIGINAL columns (easy to interpret)
    pi = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax); ax.set_title("Permutation importance (increase in MAE, lakhs)")
    fig.tight_layout(); fig.savefig(MODEL_DIR / "feature_importance.png", dpi=150); plt.close(fig)
    results["permutation_importance"] = imp.sort_values(ascending=False).round(4).to_dict()

    # ------------------------------------------------ refit on ALL data & save
    best.fit(X, y)
    joblib.dump(best, MODEL_DIR / "best_model.joblib")

    # metadata for the Streamlit app (dropdowns, coordinate defaults)
    meta = {
        "cities": sorted(df["city"].unique().tolist()),
        "localities": {c: sorted(g["locality"].unique().tolist()) for c, g in df.groupby("city")},
        "locality_stats": {
            f"{c}|{l}": {
                "latitude": float(g["latitude"].median()), "longitude": float(g["longitude"].median()),
                "distance_to_cbd_km": float(g["distance_to_cbd_km"].median()),
                "distance_to_metro_km": float(g["distance_to_metro_km"].median()),
            } for (c, l), g in df.groupby(["city", "locality"])
        },
        "furnishing": FURNISHING_ORDER,
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print("\nSaved model, metrics, plots to", MODEL_DIR)


if __name__ == "__main__":
    main()
