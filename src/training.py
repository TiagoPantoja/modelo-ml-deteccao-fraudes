import pandas as pd
import kagglehub
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from src.transformers import CustomFeatureTransformer

MODEL_PATH = "artifacts/fraud_model.joblib"


def load_data():
    print("Baixando dataset")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = f"{path}/creditcard.csv"
    return pd.read_csv(csv_path)


def train_pipeline():
    df = load_data()

    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('feature_engineering', CustomFeatureTransformer()),
        ('classifier', RandomForestClassifier(
            n_estimators=50,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    auprc = average_precision_score(y_test, y_pred_proba)
    print(f"Auprc: {auprc:.4f}")

    joblib.dump(pipeline, MODEL_PATH)


if __name__ == "__main__":
    train_pipeline()