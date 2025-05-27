# Predict on the test set
predictions = model.predict(x_test)

# Show a prediction with image
index = 0  # change this to see different predictions
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[index])} - Actual: {y_test[index]}")
plt.axis('off')
plt.show()

