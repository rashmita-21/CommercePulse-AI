
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "sales_lines.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# Each training snapshot predicts purchases in the following 3 months.
TRAIN_CUTOFFS = [
    "2010-07-01",
    "2010-10-01",
    "2011-01-01",
    "2011-04-01",
]

# The final, later period is held out for evaluation.
TEST_CUTOFF = "2011-07-01"
WINDOW_MONTHS = 3

NUMERIC_FEATURES = [
    "RecencyDays",
    "PriorInvoiceCount",
    "PriorSalesValue",
    "PriorUnits",
    "UniqueProducts",
]
CATEGORICAL_FEATURES = ["Country"]
INPUT_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_sales():
    print("Loading sales data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    sales = pd.read_csv(
        DATA_PATH,
        dtype={"Invoice": "string", "StockCode": "string"},
        low_memory=False,
    )

    required = {
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    }
    missing = required - set(sales.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    sales["InvoiceDate"] = pd.to_datetime(
        sales["InvoiceDate"], errors="coerce"
    )

    for column in ["Customer ID", "Quantity", "Price"]:
        sales[column] = pd.to_numeric(sales[column], errors="coerce")

    sales = sales.dropna(
        subset=[
            "Invoice",
            "StockCode",
            "InvoiceDate",
            "Customer ID",
            "Quantity",
            "Price",
        ]
    ).copy()

    sales = sales[
        (sales["Quantity"] > 0)
        & (sales["Price"] > 0)
        & (~sales["Invoice"].str.startswith("C", na=False))
    ].copy()

    sales["Country"] = sales["Country"].fillna("Unknown").astype(str)
    sales["LineValue"] = sales["Quantity"] * sales["Price"]

    return sales


def build_snapshot(sales, cutoff):
    """Build customer features from past data and labels from the next 3 months."""
    cutoff = pd.Timestamp(cutoff)
    target_end = cutoff + pd.DateOffset(months=WINDOW_MONTHS)

    history = sales[sales["InvoiceDate"] < cutoff]

    future = sales[
        (sales["InvoiceDate"] >= cutoff)
        & (sales["InvoiceDate"] < target_end)
    ]

    if history.empty:
        raise ValueError(f"No historical transactions before {cutoff.date()}.")

    if future.empty:
        raise ValueError(
            f"No target-period transactions from {cutoff.date()} "
            f"to {target_end.date()}."
        )

    features = history.groupby("Customer ID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        PriorInvoiceCount=("Invoice", "nunique"),
        PriorSalesValue=("LineValue", "sum"),
        PriorUnits=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Country=(
            "Country",
            lambda values: (
                values.mode().iloc[0]
                if not values.mode().empty
                else "Unknown"
            ),
        ),
    )

    features["RecencyDays"] = (
        cutoff - features["LastPurchaseDate"]
    ).dt.days

    future_customers = set(future["Customer ID"].unique())

    features["RepeatPurchase"] = (
        features.index.isin(future_customers).astype(int)
    )

    features = features.reset_index()
    features["Country"] = features["Country"].fillna("Unknown").astype(str)
    features["SnapshotDate"] = cutoff
    features["TargetPeriodEnd"] = target_end

    return features


def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_models():
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessing", make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessing", make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y_test, probabilities),
        "Average Precision": average_precision_score(
            y_test, probabilities
        ),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    sales = load_sales()
    print(f"Usable transaction rows: {len(sales):,}")

    print("\nBuilding historical training snapshots...")
    training_snapshots = []

    for cutoff in TRAIN_CUTOFFS:
        snapshot = build_snapshot(sales, cutoff)
        training_snapshots.append(snapshot)

        print(
            f"  {cutoff}: {len(snapshot):,} customers; "
            f"repeat-purchase rate = "
            f"{snapshot['RepeatPurchase'].mean():.2%}"
        )

    train = pd.concat(training_snapshots, ignore_index=True)

    print(f"\nBuilding temporal test snapshot: {TEST_CUTOFF}")
    test = build_snapshot(sales, TEST_CUTOFF)

    X_train = train[INPUT_COLUMNS]
    y_train = train["RepeatPurchase"]

    X_test = test[INPUT_COLUMNS]
    y_test = test["RepeatPurchase"]

    if y_train.nunique() != 2:
        raise ValueError("Training labels must contain both classes.")

    if y_test.nunique() != 2:
        raise ValueError(
            "Test labels contain only one class; ROC AUC cannot be calculated."
        )

    print(f"\nTraining snapshot rows: {len(train):,}")
    print(f"Later-period test customers: {len(test):,}")
    print(f"Test repeat-purchase rate: {y_test.mean():.2%}")

    models = make_models()
    fitted_models = {}
    comparison_rows = []
    reports = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        fitted_models[name] = model
        metrics = evaluate_model(model, X_test, y_test)

        comparison_rows.append(
            {
                "Model": name,
                "Training Snapshot Rows": len(train),
                "Test Customers": len(test),
                "Test Repeat-Purchase Rate": y_test.mean(),
                **metrics,
            }
        )

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        reports.append(
            f"MODEL: {name}\n"
            f"Test cutoff: {TEST_CUTOFF}\n"
            f"Target window: {TEST_CUTOFF} to "
            f"{(pd.Timestamp(TEST_CUTOFF) + pd.DateOffset(months=3)).date()}\n\n"
            + classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["No repeat purchase", "Repeat purchase"],
                zero_division=0,
            )
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "ROC AUC", ascending=False
    )

    comparison.to_csv(
        REPORTS / "ml_model_comparison.csv", index=False
    )

    best_name = comparison.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_metrics = comparison.iloc[0]

    pd.DataFrame(
        [
            {"Metric": column, "Value": best_metrics[column]}
            for column in comparison.columns
            if column != "Model"
        ]
        + [
            {"Metric": "Selected Model", "Value": best_name},
            {"Metric": "Test Cutoff", "Value": TEST_CUTOFF},
            {
                "Metric": "Test Target End",
                "Value": str(
                    (
                        pd.Timestamp(TEST_CUTOFF)
                        + pd.DateOffset(months=WINDOW_MONTHS)
                    ).date()
                ),
            },
        ]
    ).to_csv(REPORTS / "ml_model_metrics.csv", index=False)

    (REPORTS / "ml_classification_report.txt").write_text(
        "\n\n" + ("\n" + "=" * 60 + "\n\n").join(reports),
        encoding="utf-8",
    )

    # Export later-period customer scores from the selected model.
    output_columns = [
        "Customer ID",
        "LastPurchaseDate",
        "RecencyDays",
        "PriorInvoiceCount",
        "PriorSalesValue",
        "PriorUnits",
        "UniqueProducts",
        "Country",
        "RepeatPurchase",
    ]

    output = test[output_columns].copy()

    for name, model in fitted_models.items():
        safe_name = name.lower().replace(" ", "_")
        output[f"{safe_name}_probability"] = model.predict_proba(
            X_test
        )[:, 1]

    output["SelectedModel"] = best_name
    selected_probability_column = (
        best_name.lower().replace(" ", "_") + "_probability"
    )
    output["RepeatPurchaseProbability"] = output[
        selected_probability_column
    ]
    output["PredictedRepeatPurchase"] = (
        output["RepeatPurchaseProbability"] >= 0.5
    ).astype(int)
    output["PredictionCutoff"] = TEST_CUTOFF
    output["TargetPeriodEnd"] = (
        pd.Timestamp(TEST_CUTOFF)
        + pd.DateOffset(months=WINDOW_MONTHS)
    ).date()
    output = output.rename(
        columns={"RepeatPurchase": "ActualRepeatPurchase"}
    )
    output = output.sort_values(
        "RepeatPurchaseProbability", ascending=False
    )

    output.to_csv(
        REPORTS / "repeat_purchase_predictions.csv", index=False
    )

    # Compare Random Forest importance with Logistic Regression coefficients.
    rf = fitted_models["Random Forest"]
    lr = fitted_models["Logistic Regression"]

    feature_names = rf.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    rf_importance = rf.named_steps["classifier"].feature_importances_
    lr_coefficients = lr.named_steps["classifier"].coef_[0]

    pd.DataFrame(
        {
            "Feature": feature_names,
            "RandomForestImportance": rf_importance,
            "LogisticRegressionCoefficient": lr_coefficients,
            "AbsoluteLogisticCoefficient": abs(lr_coefficients),
        }
    ).sort_values(
        "RandomForestImportance", ascending=False
    ).to_csv(
        REPORTS / "ml_feature_importance.csv", index=False
    )

    # Save the selected, fitted pipeline for future reuse.
    joblib.dump(best_model, MODELS / "repeat_purchase_model.joblib")

    print("\nTEMPORAL MODEL COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSelected model by ROC AUC: {best_name}")
    print(f"Test repeat-purchase rate: {y_test.mean():.2%}")

    print("\nSaved reports:")
    print("reports/ml_model_comparison.csv")
    print("reports/ml_model_metrics.csv")
    print("reports/ml_classification_report.txt")
    print("reports/repeat_purchase_predictions.csv")
    print("reports/ml_feature_importance.csv")
    print("models/repeat_purchase_model.joblib")
    print("\nML pipeline completed.")


if __name__ == "__main__":
    main()
