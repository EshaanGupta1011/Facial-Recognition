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

# ******************************************************

# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# Flatten the feature vectors
x_train_flatten = x_train.reshape(x_train.shape[0], -1)
x_test_flatten = x_test.reshape(x_test.shape[0], -1)
#
# # Create and train the SVM classifier
# svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
# svm_classifier.fit(x_train_flatten, train['label'])
#
# # Predict labels for test data
# y_pred = svm_classifier.predict(x_test_flatten)
#
# # Calculate accuracy
# accuracy = accuracy_score(test['label'], y_pred)
# print("Accuracy:", accuracy)

from sklearn.decomposition import PCA

# Reduce the dimensionality of the feature vectors using PCA
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_flatten)

# Plot the points in 2D space
plt.figure(figsize=(10, 8))
sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=train['label'], palette='viridis')
plt.title('2D PCA Visualization of Training Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Label')
plt.show()
