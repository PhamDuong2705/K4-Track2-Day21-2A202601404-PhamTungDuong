from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tai file model.joblib tu cloud storage ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc (duoc dat trong systemd service).
    """
    # TODO 1: Tao storage.Client()
    # client = storage.Client()

    # TODO 2: Lay bucket va blob tuong ung
    # bucket = client.bucket(ARTIFACT_BUCKET)
    # blob   = bucket.blob(MODEL_KEY)

    # TODO 3: Tai file model xuong may
    # blob.download_to_filename(MODEL_PATH)

    # TODO 4: In thong bao thanh cong
    # print("Model da duoc tai xuong tu cloud storage.")

    pass  # xoa dong nay sau khi hoan thanh tat ca TODO ben tren


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    # TODO 5: Tra ve dict {"status": "ok"}
    pass  # xoa dong nay sau khi hoan thanh


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    # TODO 6: Kiem tra so luong dac trung.
    # Neu len(req.features) != 10, raise HTTPException(status_code=400, ...)

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    # pred = model.predict(...)

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    # Nhan tuong ung: 0 -> "thu_nhap_thap", 1 -> "thu_nhap_cao"
    # return {"prediction": ..., "label": ...}

    pass  # xoa dong nay sau khi hoan thanh tat ca TODO ben tren


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
