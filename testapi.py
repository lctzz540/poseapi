import requests

# Set the API endpoint URL
url = "http://localhost:8000/predict"

# Open the CSV file and read its content
with open("../Tree Pose_Right.csv", "rb") as f:
    file_content = f.read()

# Create the request payload with the file content
files = {"file": ("test_data.csv", file_content)}
print("test_data.csv", file_content)
# Send the POST request to the API endpoint with the file upload
response = requests.post(url, files=files)

# Print the response
print(response.json())
