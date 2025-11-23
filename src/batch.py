import pandas as pd
import joblib
import sys
import os

MODEL_PATH = "artifacts/fraud_model.joblib"


def batch_process(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Arquivo não encontrado {input_file}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado {MODEL_PATH}")

    print(f"Carregando modelo {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Processando {input_file}")

    chunk_size = 50000
    first_chunk = True

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        if 'Class' in chunk.columns:
            X = chunk.drop(columns=['Class'])
        else:
            X = chunk

        probabilities = model.predict_proba(X)[:, 1]

        result_df = pd.DataFrame({
            'TransactionId': chunk.index,
            'Fraud_Probability': probabilities,
            'Is_Fraud': probabilities > 0.5
        })

        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        result_df.to_csv(output_file, mode=mode, header=header, index=False)

        first_chunk = False

    print(f"Processamento salvo {output_file}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        try:
            batch_process(sys.argv[1], sys.argv[2])
        except Exception as e:
            print(f"Erro: {e}")
            sys.exit(1)
    else:
        print("Input inválido")