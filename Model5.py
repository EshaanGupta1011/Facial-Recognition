import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.cluster import KMeans


def load_dataset(directory):
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            image_path = os.path.join(directory, label, filename)
            image_paths.append(image_path)
            labels.append(label)

        print(label, "Completed")

    return image_paths, labels


def extract_features(images):
    features = []
    for image in (images):
        img = load_img(image, color_mode='grayscale', target_size=(48, 48))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


train_data_dir = "./Data/train"
test_data_dir = "./Data/test"



train = pd.DataFrame()
train['image'], train['label'] = load_dataset(train_data_dir)
# shuffle the dataset
train = train.sample(frac=1).reset_index(drop=True)

test = pd.DataFrame()
test['image'], test['label'] = load_dataset(test_data_dir)

# sns.countplot(train['label'])
# plt.show()

# for index, file, label in files.itertuples():
#     plt.subplot(5, 5, index+1)
#     img = load_img(file)
#     img = np.array(img)
#     plt.imshow(img)
#     plt.title(label)
#     plt.axis('off')
#     plt.show()

train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# *************************************************************

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn_classifier.fit(x_train.reshape(len(x_train), -1), y_train)

# Predict labels for testing data
y_pred_train = knn_classifier.predict(x_train.reshape(len(x_train), -1))
y_pred_test = knn_classifier.predict(x_test.reshape(len(x_test), -1))

# Evaluate model
train_accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_pred_train, axis=1))
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_test, axis=1))

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# *******************************************
train_accuracies = []
test_accuracies = []

k_values = range(1, 21)  # Try k values from 1 to 20

for k in k_values:
    # Train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn_classifier.fit(x_train.reshape(len(x_train), -1), y_train)

    # Predict labels for training and testing data
    y_pred_train = knn_classifier.predict(x_train.reshape(len(x_train), -1))
    y_pred_test = knn_classifier.predict(x_test.reshape(len(x_test), -1))

    # Calculate accuracy for training and testing data
    train_accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_pred_train, axis=1))
    test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_test, axis=1))

    # Append accuracies to lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(k_values, test_accuracies, label='Testing Accuracy', marker='o')
plt.title('KNN Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()