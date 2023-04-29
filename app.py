from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from fastapi.responses import JSONResponse
import cv2
import polars as pl
import mediapipe as mp


app = FastAPI()

# Load your model
model = tf.keras.models.load_model("./best_model.h5")

# Set up the Mediapipe Pose Estimator
pose_estimator = mp.solutions.pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded video file into a cv2 VideoCapture object
    video = cv2.VideoCapture(file.file)

    frame_count = 0
    df = None

    # Process each frame of the video
    while True:
        success, image = video.read()
        if not success:
            break

        frame_count += 1

        # Convert the image to RGB and run it through the pose estimator
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(image)
        if results.pose_landmarks is None:
            continue

        # Extract the pose keypoints and add them to a Polars Series
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
            keypoints.append(landmark.visibility)

        keypoints = pl.Series("keypoints", keypoints)
        new_column_name = f"keypoints_{frame_count}"
        keypoints = keypoints.rename(new_column_name)

        # Add the keypoints Series to the DataFrame
        if df is None:
            df = keypoints.to_frame()
        else:
            df = df.hstack(keypoints.to_frame())

    # Transpose the DataFrame and predict the class using your model
    try:
        df = df.transpose()
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
    except:
        return JSONResponse(content={"error": "Unable to predict class"})
