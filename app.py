from fastapi import FastAPI, UploadFile
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi.responses import JSONResponse

app = FastAPI()

# Load your model
model = tf.keras.models.load_model("./best_model.h5")


@app.post("/predict")
async def predict(file: UploadFile):
    # Read the uploaded CSV file into a Pandas DataFrame
    df = pd.read_csv(file.file)

    X = []
    no_of_timesteps = 500
    dataset = df.iloc[:, 1:].values
    n_sample = len(dataset)

    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i - no_of_timesteps: i, :])

    predictions = model.predict(np.array(X))
    label_list = []

    for i, prediction in enumerate(predictions):
        predicted_label = np.argmax(prediction)
        predicted_prob = prediction[predicted_label]
        label_list.append((predicted_label, predicted_prob))

    counts = np.bincount([label[0] for label in label_list])
    predicted_class = np.argmax(counts)

    return JSONResponse(content={"predict": int(predicted_class)})
