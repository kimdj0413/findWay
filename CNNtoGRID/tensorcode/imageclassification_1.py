import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

###########################
###     데이터 추가     ###
##########################
import pathlib
data_dir = "./flower_photos/"
# data_dir = tf.keras.utils.get_file('flower_photo', origin=dataset_url)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[0])).show()
# tulips = list(data_dir.glob('tulips/*'))
# PIL.Image.open(str(tulips[0])).show()

#################################
###     데이터 세트 만들기     ###
#################################

##  배치 사이즈와 이미지 사이즈를 지정
batch_size = 32
img_height = 180
img_width = 180

##  훈련셋 설정(8/2)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
##  검증셋 설정(8/2)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names
# print(class_names)

##############################
###     데이터 시각화하기   ###
##############################

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)    #(32, 180, 180, 3)텐서. 180x180x3 32개 이미지 묶음
#     print(labels_batch.shape)
#     break

#########################################
###     성능 높이도록 데이터세트 구성   ###
#########################################

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
##  cache() : 이미지를 메모리에 유지해 병목현상을 방지.
##  prefetch() : 데이터 전처리 및 모델 실행을 중첩.

##########################
###     데이터 표준화   ###
##########################

normalization_layer = layers.Rescaling(1./255)  #RGB 채널 값을 0~255에서 0~1로 조정

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))  #x는 이미지, y는 레이블
image_batch, labels_batch = next(iter(normalized_ds)) #iter 반복문을 통해 다음 배치를 가져와 넣음
first_image = image_batch[0]
# print(np.min(first_image), np.max(first_image))

##########################
###     모델 만들기     ###
##########################

num_classes = len(class_names)

# model = Sequential([
#     layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#     layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes)
# ])

# ##  컴파일
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.summary()

##  훈련
# epochs=10
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs   #반복 횟수
# )

##########################
###     결과 시각화     ###
##########################

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8,8))

# plt.subplot(1,2,1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validadtion Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1,2,2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validadtion Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')

# plt.show()
##  오버피팅으로 인해 검증 세트 정확도가 크게 떨어짐(오버피팅은 데이터가 작을때 발생)

##########################
###     데이터 증강     ###
##########################

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                   img_width,
                                   3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# plt.figure(figsize=(10,10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")

# plt.show()

######################
###     드롭아웃    ###
######################

##  과적합을 줄이는 또 다른 방법.
##  소수점으로 설정
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),    #출력단위의 20%를 임의로 제거
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())
###     가중치 체크 포인트 저장
# checkpoint_path = "c:/waytoMap/cp.weights.h5"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint (
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     verbose=1
# )

# model.load_weights(checkpoint_path)

epochs = 15
# with tf.device("/gpu:0"):
history = model.fit(
    train_ds
    , validation_data=val_ds
    , epochs=epochs
    # , callbacks=[cp_callback]
)

##      모델저장
model.save('imageclassfication.keras')

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8,8))

# plt.subplot(1,2,1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validadtion Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1,2,2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validadtion Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')

# plt.show()

#################################
###     새로운 데이터 예측      ###
#################################
##      학습 모델 저장 후 다른 페이지에서 실행
# model = keras.models.load_model('imageclassfication.keras')
# sunflower_path = "c:/Red_sunflower.jpg"
# img = tf.keras.utils.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )