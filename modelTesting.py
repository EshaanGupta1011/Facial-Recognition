import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

model = load_model('./Model/model.h5')  # Load your trained model

# Step 2: Load your test data
test_data_path = "./Data/test"
test_filenames = os.listdir(test_data_path)

# Assuming your test data is images, and you're using ImageDataGenerator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(48,48),  # specify your image dimensions
    batch_size=16,
    class_mode='categorical',  # or specify the appropriate class mode
    shuffle=False
)

# Step 4: Use the loaded model to make predictions on the test data
predictions = model.predict(test_generator)

# Step 5: Compare the predicted labels with the true labels from the test data
true_labels = test_generator.classes
predicted_labels = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_labels == true_labels)
print("Accuracy:", accuracy)
