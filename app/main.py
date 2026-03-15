from fastapi import FastAPI, UploadFile, File
import pandas as pd
import shutil

from pipeline import train_pipeline, predict_pipeline

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML Pipeline API Running"}

@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):

    dataset_path = f"datasets/{file.filename}"

    with open(dataset_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    message = train_pipeline(dataset_path)

    return {"status": message}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    predictions = predict_pipeline(df)

    return {"predictions": predictions}
