import os
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

# JSON ve görsel dosyalarının yolları
json_dir = r'C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\labels'
image_dir = r'C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\images'

# Dosyaları listele
json_files = sorted([os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')])
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])


def create_mask_from_json(json_path, image_shape):
    with open(json_path, 'r') as file:
        data = json.load(file)

    mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(mask)

    for annotation in data['annotations']:
        segmentation = annotation['segmentation']
        for segment in segmentation:
            draw.polygon(segment, outline=None, fill=1)

    return img_to_array(mask)


images = []
masks = []

for image_path, json_path in zip(image_files, json_files):
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image) / 255.0
    images.append(image)

    mask = create_mask_from_json(json_path, (256, 256))
    masks.append(mask)

images = np.array(images)
masks = np.array(masks)

# Maske boyutunu sıkıştır
masks = np.squeeze(masks, axis=-1)
masks = np.expand_dims(masks, axis=-1)
masks = masks.astype(np.float32)


print(f'Görüntüler şekli: {images.shape}')
print(f'Maskeler şekli (düzeltilmiş): {masks.shape}')

# Verileri eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.1, random_state=42)
import matplotlib.pyplot as plt

# Eğitim verilerinden birkaç örnek gösterelim
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i])
    plt.axis('off')
    plt.title('Görüntü')

    plt.subplot(2, 5, i+6)
    plt.imshow(y_train[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Maske')
plt.show()


# UNet modeli
def unet(input_size=(256, 256, 3)):
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

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


model = unet()
model.summary()

# UNet modelini eğit
history = model.fit(X_train, y_train, batch_size=8, epochs=2, verbose=1, validation_data=(X_test, y_test))

# Tahmin edilen maskeleri görselleştir
predicted_mask = model.predict(X_test)

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1)
    plt.title('Orijinal Görüntü')
    plt.imshow(X_test[i])

    plt.subplot(3, 3, i * 3 + 2)
    plt.title('Gerçek Maske')
    plt.imshow(y_test[i].squeeze(), cmap='gray')

    plt.subplot(3, 3, i * 3 + 3)
    plt.title('Tahmin Edilen Maske')
    plt.imshow(predicted_mask[i].squeeze(), cmap='gray')

plt.show()

# Eğitim geçmişini görselleştir
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.legend()
plt.title('Eğitim ve Doğrulama Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
plt.legend()
plt.title('Eğitim ve Doğrulama Accuracy')
plt.show()
# Eğitim geçmişini görselleştir
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.legend()
plt.title('Eğitim ve Doğrulama Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
plt.legend()
plt.title('Eğitim ve Doğrulama Accuracy')
plt.show()

predicted_masks = model.predict(X_test)

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title('Orijinal Görüntü')

    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(y_test[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Gerçek Maske')

    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(predicted_masks[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Tahmin Edilen Maske')
plt.show()
