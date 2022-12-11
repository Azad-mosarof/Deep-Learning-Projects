import random
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.layers import Input,Conv2D,Lambda,Dropout,MaxPool2D
from keras.layers import Conv2DTranspose,Concatenate
from keras import Model
import os
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
import numpy as np
import matplotlib.pyplot as plt

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNEL = 3

seed = 40
np.random.seed = seed

train_path = "/home/azadm/Desktop/Datasetf_For_ML/images_of_neuclie/stage1_train"
test_path = "/home/azadm/Desktop/Datasetf_For_ML/images_of_neuclie/stage1_test"

train_ids = next(os.walk(train_path))[1]
test_ids = next(os.walk(test_path))[1]

#train images and masks
X_train = np.zeros((len(train_ids), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)

print("Resizing training image and masks")
for n, id in tqdm(enumerate(train_ids),total=len(train_ids)):
    path = train_path + '/' + id
    img = imread(path + '/images/' + id + '.png')
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), mode="constant", preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMAGE_HEIGHT, IMAGE_WIDTH), 
                            mode="constant", preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

#test images
X_test = np.zeros((len(test_ids), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype=np.uint8)
print("Resizing test images")
sizes_test = []
for n, id in tqdm(enumerate(test_ids),total=len(test_ids)):
    path = test_path + '/' + id
    img = imread(path + '/images/' + id + '.png')
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), mode="constant", preserve_range=True)
    X_test[n] = img

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()

input_layer = Input((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL))
s = Lambda(lambda x: x/255)(input_layer)

conv_1 = Conv2D(16, (3,3), 
            activation='relu',
            padding='same', 
            kernel_initializer='he_normal')(s)
conv_1 = Dropout(0.1)(conv_1)
conv_1 = Conv2D(16, (3,3), 
            activation='relu',
            padding='same', 
            kernel_initializer='he_normal')(conv_1)
p1 = MaxPool2D(pool_size=(2,2))(conv_1)
    
conv_2 = Conv2D(32, (3,3), 
            activation='relu',
            padding='same', 
            kernel_initializer='he_normal')(p1)
conv_2 = Dropout(0.1)(conv_2)
conv_2 = Conv2D(32, (3,3), 
            activation='relu',
            padding='same', 
            kernel_initializer='he_normal')(conv_2)
p2 = MaxPool2D(pool_size=(2,2))(conv_2)

conv_3 = Conv2D(64, (3,3), 
            activation='relu',
            padding='same', 
            kernel_initializer='he_normal')(p2)
conv_3 = Dropout(0.1)(conv_3)
conv_3 = Conv2D(64, (3,3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal')(conv_3)
p3 = MaxPool2D(pool_size=(2,2))(conv_3)

conv_4 = Conv2D(128, (3,3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal')(p3)
conv_4 = Dropout(0.2)(conv_4)
conv_4 = Conv2D(128, (3,3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal')(conv_4)
p4 = MaxPool2D(pool_size=(2,2))(conv_4)

conv_5 = Conv2D(256, (3,3), 
            activation='relu',
            padding='same', 
            kernel_initializer='he_normal')(p4)
conv_5 = Dropout(0.3)(conv_5)
conv_5 = Conv2D(256, (3,3), 
            activation='relu',
            padding='same', 
            kernel_initializer='he_normal')(conv_5)

#Expansion path
u6 = Conv2DTranspose(128, (2,2), 
            strides=(2, 2), 
            padding='same')(conv_5)
u6 = Concatenate()([u6, conv_4])
c6 = Conv2D(128, (3,3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same')(c6)

u7 = Conv2DTranspose(64, (2,2), 
            strides=(2, 2), 
            padding='same')(c6)
u7 = Concatenate()([u7, conv_3])
c7 = Conv2D(64, (3,3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same')(c7)

u8 = Conv2DTranspose(32, (2,2), 
            strides=(2, 2), 
            padding='same')(c7)
u8 = Concatenate()([u8, conv_2])
c8 = Conv2D(32, (3,3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same')(u8)
c8 = Dropout(0.2)(c8)
c8 = Conv2D(32, (3, 3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same')(c8)

u9 = Conv2DTranspose(16, (2,2), 
            strides=(2, 2), 
            padding='same')(c8)
u9 = Concatenate()([u9, conv_1])
c9 = Conv2D(16, (3,3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same')(u9)
c9 = Dropout(0.2)(c9)
c9 = Conv2D(16, (3, 3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same')(c9)

output_layer = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

## Model Checkpoint
checkpoint = ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        EarlyStopping(patience=3, monitor='val_loss'),
        TensorBoard(log_dir='logs')
]

model.load_weights('model_for_nuclei.h5')
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=16, epochs=25, callbacks=callbacks)
model.save('model_for_nuclei.h5')

## model prediction
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

#perform a sanity check on some random trainnig samples
idx = random.randint(0, len(preds_train_t))
imshow(X_train[idx])
plt.show()
imshow(Y_train[idx])
plt.show()
imshow(np.squeeze(preds_train_t[idx]))
plt.show()

#perform a sanity check on some random validation samples
idx = random.randint(0, len(preds_val_t))
imshow(X_train[idx])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][idx]))
plt.show()
imshow(np.squeeze(preds_val_t[idx]))
plt.show()

#perform a sanity check on some random test samples
idx = random.randint(0, len(preds_test_t)-1)
imshow(X_train[idx])
plt.show()
imshow(np.squeeze(preds_test_t[idx]))
plt.show()