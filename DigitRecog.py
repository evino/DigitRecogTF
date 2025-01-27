import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = np.array(train_images)


# print(train_images[1] / 255.0)
# print(train_labels[1])

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Preprocess Image
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

print(train_images.shape)
print(train_labels.shape)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Build Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")


# Select an image from the test set (e.g., the first image)
# batch_images = test_images[:5]

# # batch_images = batch_images.astype(np.float32)
# print(batch_images.shape)  # Should print (1, 28, 28, 1)
# print(batch_images.dtype)  # Should print float32 after normalization


# Predict the label for this image
predictions = model.predict(test_images[:5])

# Print the raw predictions (probabilities for each class)
print("Raw predictions for the first image:")
print(predictions[0])  # Output the raw probabilities for the first image

# Get the predicted labels (index of highest probability for each image in the batch)
predicted_labels = np.argmax(predictions, axis=1)

# Print the predictions
print("Predicted labels:", predicted_labels)

# Optionally, display the images with the predicted labels
for i in range(5):
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f"Label: {test_labels[i]}")

    # Save the image to a file
    img_path = f'./predicted_image_{predicted_labels[i]}.png'
    plt.savefig(img_path)

    # Display in the browser
    display(Image(filename=img_path))


# # # Plot the image using matplotlib
