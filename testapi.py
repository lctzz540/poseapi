import requests

# Set the API endpoint URL
url = "http://localhost:8000/predict"

# Open the video file and read its content
with open("../Tree Pose_Right.mp4", "rb") as f:
    file_content = f.read()

# Create the request payload with the file content
files = {"file": ("test_data.mp4", file_content)}

# Send a POST request to the API endpoint with the payload
response = requests.post(url, files=files)

# Print the JSON response
print(response.json())
