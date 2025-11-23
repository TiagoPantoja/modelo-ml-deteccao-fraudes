from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os

MODEL_PATH = "artifacts/fraud_model.joblib"
EXPECTED_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        ml_models["fraud_model"] = joblib.load(MODEL_PATH)
        print("Modelo carregado")
    else:
        print(f"Modelo não encontrado {MODEL_PATH}.")
    yield
    ml_models.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="API detecção de fraude com Scikit-Learn",
    lifespan=lifespan
)


class TransactionInput(BaseModel):
    Time: float = Field(..., description="Segundos desde a primeira transação")
    Amount: float = Field(..., description="Valor da transação")
    V_features: dict = Field(..., description="Dicionário V1 a V28")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 0,
                "Amount": 100.0,
                "V_features": {f"V{i}": 0.5 for i in range(1, 29)}
            }
        }
    )


class PredictionOutput(BaseModel):
    is_fraud: bool
    fraud_probability: float


@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    model = ml_models.get("fraud_model")

    if not model:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        data = transaction.V_features.copy()
        data['Time'] = transaction.Time
        data['Amount'] = transaction.Amount

        input_df = pd.DataFrame([data])

        missing_cols = set(EXPECTED_COLUMNS) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Colunas faltando {missing_cols}")

        input_df = input_df[EXPECTED_COLUMNS]

        probability = model.predict_proba(input_df)[0][1]
        prediction = probability > 0.5

        return {
            "is_fraud": bool(prediction),
            "fraud_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)