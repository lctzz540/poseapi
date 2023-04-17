# Use an official Go runtime as a parent image
FROM golang:1.16-alpine

# Install Git and other dependencies
RUN apk update && apk add --no-cache git

# Set the working directory inside the container
WORKDIR /app

# Copy the Go module files into the container
COPY go.mod go.sum ./

# Download and install the Go dependencies
RUN go mod download

# Copy the source code into the container
COPY . .

# Build the application
RUN go build -o main .

# Start a new container from the official TensorFlow image
FROM tensorflow/tensorflow:latest-gpu

# Copy the application binary from the first container to the second container
COPY --from=0 /app/main .

# Set the command to run the application
CMD ["./main"]
