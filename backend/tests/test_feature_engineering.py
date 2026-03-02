from app.services.feature_engineering import build_feature_matrix
import pandas as pd


def test_build_feature_matrix_basic():
    df = pd.DataFrame(
        [
            {
                "TransactionAmt": 100.0,
                "TransactionDT": 3600,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com",
                "DeviceType": "desktop",
                "DeviceInfo": "Windows",
            }
        ]
    )

    X, numeric, categorical = build_feature_matrix(df)

    assert "TransactionAmt" in numeric
    assert "TransactionAmt_log" in X.columns
    assert "P_emaildomain_group" in X.columns
    assert len(X) == 1

