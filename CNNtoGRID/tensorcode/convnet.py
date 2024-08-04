import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

##############################
###     데이터 전처리       ###
##############################

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin = _URL, extract = True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

class_names = train_dataset.class_names

# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

# plt.show()

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

##      성능 향상 데이터세트 구성
##      버퍼링된 프리패치를 사용해 I/O 차단없이 디스크에 이미지 로드.
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

##      데이터 증강
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2)
])

# for image, _ in train_dataset.take(1):
#     plt.figure(figsize=(10,10))
#     first_image = image[0]
#     for i in range(9):
#         ax = plt.subplot(3,3,i+1)
#         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#         plt.imshow(augmented_image[0]/255)
#         plt.axis('off')
# plt.show()

##      픽셀 값 재조정([0,255]에서 [-1,1]로)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

##      사전 훈련된 CNN으로부터 기본 모델 생성
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
##      (160, 160, 3)   ->  (32, 5, 5, 1280)

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
# print(feature_batch.shape)

##########################
###     특징 추출       ###
##########################

##      CNN의 가중치가 훈련 중 업데이트 되는 것을 방지.
base_model.trainalbe = False
# base_model.summary()
##      많은 모델에는 BatchNormalization 레이어가 포함되어 있는데 기본 모델을 호출할때 training = False를 전달해 레이어를 추론 모드로 유지해야 한다. 그렇지 않으면 모델이 학습한 내용 파괴.

##      이미지당 하나의 1280 요소 벡터로 변환해 5X5 공간 위치에 대한 평균 구함
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)
##      (32, 1280)

##      이미지당 단일 예측으로 변환
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)
##      (32, 1)

inputs = tf.keras.Input(shape=(160,160,3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training = False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

##########################
###     모델 컴파일     ###
##########################

base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.Adam              (learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
##      BinaryCrossentropy : 이진분류 문제에서 사용
##                           활성 함수는 sigmoid
##      CategoricalCrossentropy : 다중 분류에서 사용
##                                label이 벡터 값을 가질 때 사용. ex) [1,0,0]
##                                활성 함수는 softmax 함수
##      SparseCategoricalCrossentropy : 다중 분류에서 사용
##                                      label이 class index 값을 가질때 사용 ex) 0 or 1 or 2
##                                      활성 함수는 softmax 함수

##########################
###     모델 훈련       ###
##########################

initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)
# print("initial loss: {:.2f}".format(loss0))
# print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs = initial_epochs,
                    validation_data = validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

##########################
###     미세조정        ###
##########################

##      사전의 훈련된 모델의 성능을 향상 시키는 방법 -> 미세조정
##      미세조정은 최상위 층(코드 아래 부분)을 애상으로 함.
base_model.trainable = True
# print("Number of layers in the base model: ", len(base_model.layers))
##      Number of layers in the base model:  154
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False