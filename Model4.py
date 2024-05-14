from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img


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


# *****************************************************

naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)

y_pred = naive_bayes.predict(x_test)
x_predict = naive_bayes.predict(x_train)

x_predict_labels = np.argmax(naive_bayes.predict_proba(x_train), axis=1)

accuracy_train = accuracy_score(y_train, x_predict_labels)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy of training data:", accuracy_train)
