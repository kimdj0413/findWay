import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
# _ = plt.imshow(image)
# _ = plt.title(get_label_name(label))
# plt.show()

##################################
###     keras 전처리 레이어     ###
##################################

##      크기 및 배율 조정하기
IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])
##      180x180, [0,1] 범위로 표준화
##      [-1, 1] 원하면 tf.keras.layers.Rescaling(1./127.5, offset=-1)
# result = resize_and_rescale(image)
# _ = plt.imshow(result)
# plt.show()
# print("Min and max pixel values:", result.numpy().min(), result.numpy().max())
##      픽셀이 [0, 1] 범위에 있는지 확인

##      데이터 증강
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])
##      이외에도 layers.RandomContrast, layers.RandomCrop, layers.RandomZoom 등 전처리 레이어

image = tf.cast(tf.expand_dims(image, 0), tf.float32)

# plt.figure(figsize=(10,10))
# for i in range(9):
#     augmented_image = data_augmentation(image)
#     ax = plt.subplot(3, 3, i+1)
#     plt.imshow(augmented_image[0])
#     plt.axis("off")
# plt.show()

##      사용법 #1 : 전처리 레이어를 모델의 일부로 만들기
# model = tf.keras.Sequential([
#     resize_and_rescale,
#     data_augmentation,
#     layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D()
# ])

##      사용법 #2 : 데이터세트에 전처리 레이어 적용
# aug_ds = train_ds.map(
#     lambda x, y: (resize_and_rescale(x, training=True), y)
# )

##      데이터 증강은 '훈련 세트'에만 적용해야 한다.

##      데이터 세트에 전처리 레이어 적용
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False, augment=False):
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

# ##########################
# ###     모델 훈련       ###
# ##########################

# model = tf.keras.Sequential([
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# epochs=5
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# loss, acc = model.evaluate(test_ds)
# print("Accuracy", acc)

###     사용자 정의는 하지 않음     ###

##################################
###     tf.image 사용하기       ###
##################################
##      keras 전처리 유틸리티는 편하지만 tf.image를 통해 세밀한 제어가 가능

##      이미지 다시 불러오기
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

image, label = next(iter(train_ds))
# _ = plt.imshow(image)
# _ = plt.title(get_label_name(label))
# plt.show()

##      시각화 함수
def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()

##      이미지 뒤집기
flipped = tf.image.flip_left_right(image)
# visualize(image, flipped)

##      이미지 그레이스케일
grayscaled = tf.image.rgb_to_grayscale(image)
# visualize(image, tf.squeeze(grayscaled))

##      이미지 포화시키기
satured = tf.image.adjust_saturation(image, 3)
# visualize(image, satured)

##      이미지 가운데 자르기
cropped = tf.image.central_crop(image, central_fraction=0.5)
# visualize(image, cropped)

##      이미지 회전
rotated = tf.image.rot90(image)
# visualize(image, rotated)

##########################
###     무작위 변환     ###
##########################
##      tf.image.random 및 tf.image.stateless_random 사용
##      주로 tf.image.stateless_random 사용(TF 2.X 버전용)

##      무작위 이미지 밝기 변경
for i in range(3):
  seed = (i, 0)  # tuple of size (2,)
  stateless_random_brightness = tf.image.stateless_random_brightness(
      image, max_delta=0.95, seed=seed)
#   visualize(image, stateless_random_brightness)
##      밝기 계수[-max_delta, max_delta]

##      무작위 대비 변경
for i in range(3):
  seed = (i, 0)  # tuple of size (2,)
  stateless_random_contrast = tf.image.stateless_random_contrast(
      image, lower=0.1, upper=0.9, seed=seed)
#   visualize(image, stateless_random_contrast)
##      대비 범위[lower, upper]

##      무작위 자르기
for i in range(3):
  seed = (i, 0)  # tuple of size (2,)
  stateless_random_crop = tf.image.stateless_random_crop(
      image, size=[210, 300, 3], seed=seed)
  visualize(image, stateless_random_crop)

# tf.image.stateless_random_brightness
# tf.image.stateless_random_contrast
# tf.image.stateless_random_crop
# tf.image.stateless_random_flip_left_right
# tf.image.stateless_random_flip_up_down
# tf.image.stateless_random_hue
# tf.image.stateless_random_jpeg_quality
# tf.image.stateless_random_saturation

##      데이터 세트에 적용하기
##      데이터 생성
(train_datasets, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

##      이미지 크기 조정
def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label

##      랜덤 함수 생성
def augment(image_label, seed):
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  ##        시드생성
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  ##        랜덤 크롭
  image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  ##        랜덤 밝기
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label

# Create a `Counter` object and `Dataset.zip` it together with the training set.
counter = tf.data.experimental.Counter()
train_ds = tf.data.Dataset.zip((train_datasets, (counter, counter)))

train_ds = (
    train_ds
    .shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)