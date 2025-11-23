import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.batch import batch_process

@patch("src.batch.joblib.load")
@patch("src.batch.os.path.exists")
def test_batch_success(mock_exists, mock_load, tmp_path):
    mock_exists.return_value = True

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    mock_load.return_value = mock_model

    input_csv = tmp_path / "test_input.csv"
    output_csv = tmp_path / "test_output.csv"

    cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    df = pd.DataFrame(0.5, index=[0, 1], columns=cols)
    df.to_csv(input_csv, index=False)

    batch_process(str(input_csv), str(output_csv))

    assert output_csv.exists()

    df_res = pd.read_csv(output_csv)
    assert len(df_res) == 2
    assert 'Fraud_Probability' in df_res.columns
    assert 'Is_Fraud' in df_res.columns
    assert df_res['Fraud_Probability'].iloc[0] == 0.9

def test_batch_input_file_not_found():
    with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
        batch_process("arquivo_fantasma.csv", "saida.csv")

@patch("src.batch.os.path.exists")
def test_batch_model_not_found(mock_exists):
    def side_effect(path):
        if "input.csv" in str(path): return True
        return False

    mock_exists.side_effect = side_effect

    with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
        batch_process("input.csv", "output.csv")