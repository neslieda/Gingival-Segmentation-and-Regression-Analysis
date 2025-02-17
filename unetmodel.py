import os
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

json_path = r'C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\labels\IMG_2516.json'
image_path = r'C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\images\IMG_2516.jpg'

with open(json_path, 'r') as file:
    data = json.load(file)

annotations = data['annotations'][0]['segmentation'][0]

image = load_img(image_path)
image = img_to_array(image)
mask = np.zeros((image.shape[0], image.shape[1]))

points = np.array(annotations).reshape((-1, 2))
points = points.astype(np.int32)
cv2.fillPoly(mask, [points], 1)

# Resimleri normalize edelim
image = image / 255.0
mask = np.expand_dims(mask, axis=-1)

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = unet_model()

from sklearn.model_selection import train_test_split

# Görüntüleri ve maskeleri eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(image, mask, test_size=0.1, random_state=42)

# Modeli eğit
history = model.fit(X_train, y_train, batch_size=1, epochs=50, verbose=1, validation_data=(X_test, y_test))

predicted_mask = model.predict(np.expand_dims(X_test[0], axis=0))

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Orijinal Görüntü')
plt.imshow(X_test[0])

plt.subplot(1, 3, 2)
plt.title('Gerçek Maske')
plt.imshow(y_test[0].squeeze(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Tahmin Edilen Maske')
plt.imshow(predicted_mask[0].squeeze(), cmap='gray')

plt.show()
