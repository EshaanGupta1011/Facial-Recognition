from sklearn.tree import DecisionTreeClassifier, plot_tree
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

train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

# Use integer-encoded labels for training
decision_tree = DecisionTreeClassifier(max_depth=50, random_state=42)
decision_tree.fit(train_features.reshape(len(train_features), -1), y_train)

# Evaluate the decision tree model
train_accuracy = decision_tree.score(train_features.reshape(len(train_features), -1), y_train)
test_accuracy = decision_tree.score(test_features.reshape(len(test_features), -1), y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(decision_tree, filled=True, feature_names=[str(i) for i in range(48*48)], class_names=le.classes_)
plt.show()
