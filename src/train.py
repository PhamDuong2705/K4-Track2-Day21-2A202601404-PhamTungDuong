import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """
    # 1 load data
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2: Tach dac trung (X) va nhan (y)
    x_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    x_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"] 

    # 3 MLflow: ghi nhan cac sieu tham so
    with mlflow.start_run():

        # hyperparameters = params["hyperparameters"]
        mlflow.log_params(params)

        #train 
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(x_train, y_train)

        #evaluate
        preds = model.predict(x_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        #log LMflow metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # MLflow model artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # CI report
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/ci_report.txt", "w") as f:
            json.dump({"f1": f1, "accuracy": acc}, f, indent=2)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

        os.makedirs("outputs", exist_ok=True)

        report = {
            "f1_score": float(f1),
            "accuracy": float(acc),
        }

        with open("outputs/report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
