import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from model import create_model

MODEL_PATH = "model.pkl"

def train_pipeline(dataset_path):

    df = pd.read_csv(dataset_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = create_model()

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    return "Model trained successfully"


def predict_pipeline(df):

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(df)

    return predictions.tolist()
