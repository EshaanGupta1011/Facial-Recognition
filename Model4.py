from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical


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
    for image in images:
        img = load_img(image, color_mode='grayscale', target_size=(48, 48))
        img = np.array(img).flatten()
        features.append(img)
    return np.array(features)


train_data_dir = "./Data/train"
test_data_dir = "./Data/test"

train = pd.DataFrame()
train['image'], train['label'] = load_dataset(train_data_dir)
# shuffle the dataset
train = train.sample(frac=1).reset_index(drop=True)

test = pd.DataFrame()
test['image'], test['label'] = load_dataset(test_data_dir)

train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_test = le.transform(test['label'])


# Initialize and train Naive Bayes classifier
naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)

# Predict labels for testing data
y_pred = naive_bayes.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)