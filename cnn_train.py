import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np

np.random.seed(1337)
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(16, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(8, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(classifier.summary())

train_dir = '/home/vignesh/tomato_data/train'
val_dir = '/home/vignesh/tomato_data/val'

train_data_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)
class_names = train_data_raw.class_names  # Get class names before mapping
train_data = train_data_raw.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)
val_data = val_data.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

print({name: idx for idx, name in enumerate(class_names)})

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

classifier.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
)

classifier.save('keras_potato_trained_model(2.h5')
print('Saved trained model as %s ' % 'keras_potato_trained_model.h5')
