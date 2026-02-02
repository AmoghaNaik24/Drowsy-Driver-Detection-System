import os
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Generator
def generator(dir, gen=ImageDataGenerator(rescale=1./255), batch_size=32, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

# Paths for training and validation datasets
train_batch = generator('data/train', batch_size=32)
valid_batch = generator('data/valid', batch_size=32)

# Model Definition
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=len(train_batch), validation_steps=len(valid_batch))

# Save Model
os.makedirs('models', exist_ok=True)
model.save('models/cnnCat2.h5', save_format='h5')
print("Model saved successfully!")
