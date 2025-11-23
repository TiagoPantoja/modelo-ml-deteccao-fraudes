import pandas as pd
from src.transformers import CustomFeatureTransformer, binary_function_x_plus_1


def test_binary_function_logic():
    assert binary_function_x_plus_1(10) == 11
    assert binary_function_x_plus_1(-1) == 0

def test_transformer_integrates_logic():
    df = pd.DataFrame({'Amount': [100.0, 200.0], 'V1': [1, 2]})

    transformer = CustomFeatureTransformer()
    df_transformed = transformer.transform(df)

    assert 'Amount_Plus_One' in df_transformed.columns
    assert df_transformed['Amount_Plus_One'].iloc[0] == 101.0
    assert df_transformed['V1'].iloc[0] == 1


def test_transformer_immutability():
    df = pd.DataFrame({'Amount': [10.0]})
    df_original = df.copy()

    transformer = CustomFeatureTransformer()
    _ = transformer.transform(df)

    pd.testing.assert_frame_equal(df, df_original)


def test_transformer_missing_column():
    df = pd.DataFrame({'V1': [1, 2]})  # Sem Amount
    transformer = CustomFeatureTransformer()

    res = transformer.transform(df)

    assert 'Amount_Plus_One' not in res.columns
    assert res.shape == (2, 1)