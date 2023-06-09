
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt -v

# Install additional packages for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Copy the rest of the application code into the container at /app
COPY . .

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Start the FastAPI application using uvicorn
CMD ["streamlit", "run", "app.py"]
