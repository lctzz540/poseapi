import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data shape for the model
input_shape = (500, 16)

# Load the trained TensorFlow model
model = tf.keras.models.load_model("./best_model.h5.h5")

# Define the FastAPI application
app = FastAPI()

# Define the request payload as a Pydantic model


class PredictionRequest(BaseModel):
    data: list[list[float]]


# Define the response payload as a Pydantic model
class PredictionResponse(BaseModel):
    class_name: str
    confidence: float


# Define the endpoint for making predictions
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Convert the input data to a NumPy array
    x = tf.keras.preprocessing.sequence.pad_sequences(
        request.data, maxlen=input_shape[0], dtype="float32", padding="post"
    )

    # Make a prediction with the model
    prediction = model.predict(x)[0]

    # Get the predicted class and confidence score
    predicted_class = tf.argmax(prediction).numpy()
    confidence = prediction[predicted_class]

    # Map the predicted class index to a class name (if applicable)
    class_name = str(predicted_class)

    # Return the prediction result
    return PredictionResponse(class_name=class_name, confidence=confidence)
