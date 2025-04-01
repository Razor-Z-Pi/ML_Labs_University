import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist

ds = tfds.load('mnist', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)

ds = tfds.load('mnist', split='train', as_supervised=True)
ds = ds.take(1)

for image, label in ds: 
  print(image.shape, label)


builder = tfds.builder('mnist')

builder.download_and_prepare()

ds = builder.as_dataset(split='train', shuffle_files=True)
print(ds)


ds, info = tfds.load('mnist', split='train', with_info=True)

fig = tfds.show_examples(ds, info)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

plt.imshow(X_train[12], cmap = 'binary')
plt.axis('off')
print(y_train[12])

X_train = X_train / 255
X_test = X_test / 255

print(y_train[0])

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(y_train[0])

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(X_train[0].shape)))

model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Flatten())

model.add(Dense(10, activation='sigmoid'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5 )

k = 6
plt.imshow(X_test[k], cmap = 'binary')
plt.axis('off')
print(y_test[k])

k = 10
plt.imshow(X_test[k], cmap='binary')
plt.axis('off')
print(y_test[k])