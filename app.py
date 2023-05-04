import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import streamlit as st
import time
import tensorflow as tf

model = tf.keras.models.load_model("./best_model.h5")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

st.set_page_config(layout="wide")
col1, col2 = st.columns(2)
with col1:
    st.header("Pose Detection")
    image_placeholder = st.empty()
with col2:
    st.header("Key Points")
    keypoint_placeholder = st.empty()

keypoints_list = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    record_keypoints = False
    record_button = st.button("Record")

    if record_button:
        for i in range(5, 0, -1):
            st.write(i)
            time.sleep(1)
        record_keypoints = True

    start_time = time.time()
    while time.time() - start_time < 30 and record_keypoints:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame)

        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        image_placeholder.image(annotated_image, use_column_width=True)

        keypoints = []
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints.extend(
                [landmark.x, landmark.y, landmark.z, landmark.visibility])

        # Only store the first 300 keypoints
        keypoints = keypoints[:1200]

        # Append the keypoints to the keypoints_list
        keypoints_list.append(keypoints)

        if cv2.waitKey(1) == 27:
            break

        if len(keypoints_list) > 0:
            df = pd.DataFrame(keypoints_list)
            df.columns = [f"keypoints_{i}" for i in range(df.shape[1])]
            keypoint_placeholder.table(df)

    if record_keypoints:
        X = []

        no_of_timesteps = 50

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

        predicted_probs = [
            label[1] for label in label_list if label[0] == predicted_class
        ]
        confidence = np.mean(predicted_probs)

        if confidence >= 0.7:
            pose_label = {0: "Tree Pose", 1: "Warrior 1", 2: "Warrior 2"}.get(
                predicted_class, "Unknown"
            )
            st.write("Predicted pose:", pose_label)
            st.write("Confidence:", confidence)
        else:
            st.write("Incorrect")
cap.release()
cv2.destroyAllWindows()
