package main

import (
	"fmt"
	"log"
	"net/http"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/gofiber/fiber/v2"
)

func main() {
	model, err := tf.LoadSavedModel("./best_model.h5", []string{"serve"}, nil)
	if err != nil {
		log.Fatalf("Failed to load the model: %v", err)
	}

	// Create a new Fiber app
	app := fiber.New()

	// Define the prediction endpoint
	app.Post("/predict", func(c *fiber.Ctx) error {
		// Parse the input data
		inputs := make([]float32, 500*<number_of_features>)
		if err := c.BodyParser(&inputs); err != nil {
			return c.Status(http.StatusBadRequest).JSON(fiber.Map{
				"error": "Failed to parse input data",
			})
		}

		// Reshape the input data to match the model's input shape
		inputTensor, err := tf.NewTensor(inputs, tf.Shape{1, 500, <number_of_features>})
		if err != nil {
			return c.Status(http.StatusBadRequest).JSON(fiber.Map{
				"error": "Failed to reshape input data",
			})
		}

		// Run the prediction using the loaded model
		outputs, err := model.Session.Run(
			map[tf.Output]*tf.Tensor{
				model.Graph.Operation("serving_default_input_1").Output(0): inputTensor,
			},
			[]tf.Output{
				model.Graph.Operation("StatefulPartitionedCall").Output(0),
			},
			nil,
		)
		if err != nil {
			return c.Status(http.StatusInternalServerError).JSON(fiber.Map{
				"error": "Failed to run the prediction",
			})
		}

		// Extract the predicted class and confidence from the output tensor
		predictions := outputs[0].Value().([][]float32)
		predictedClass := argmax(predictions[0])
		confidence := predictions[0][predictedClass]

		// Return the predicted class and confidence as a JSON response
		return c.JSON(fiber.Map{
			"predicted_class": predictedClass,
			"confidence": confidence,
		})
	})

	// Start the HTTP server
	log.Fatal(app.Listen(":3000"))
}

func argmax(array []float32) int {
	maxIndex := 0
	maxValue := array[maxIndex]
	for i, value := range array {
		if value > maxValue {
			maxIndex = i
			maxValue = value
		}
	}
	return maxIndex
}
