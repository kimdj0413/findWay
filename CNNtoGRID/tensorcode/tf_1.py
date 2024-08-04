import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

##########################################
###     패션 MNIST 데이터 가져오기      ###
##########################################

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(fashion_mnist.load_data())

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##  레이블과 클래스로 묶여있어서 클래스 이름이 없으므로 별도 지정

##########################
###     데이터 탐색     ###
##########################

# print(train_images.shape)     #60000개의 이미지 28X28 픽셀
# print(len(train_labels))      #60000개의 이미지
# print(train_labels)           #레이블들 살펴보기
# print(test_images.shape)      #테스트 세트에는 10000개의 이미지 28X28
# print(len(test_labels))       #테스트세트 10000개의 이미지

##############################
###     데이터 전처리       ###
##############################

##  첫번째 이미지를 살펴보면 픽셀 범위가 0~255 사이.
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

##  훈련세트와 테스트세트 값의 범위를 0~1 사이로 조절
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)    #여러개를 하나의 그래프로(row, col, index)
#     plt.xticks([])          #x축 눈금 설정
#     plt.yticks([])          #y축 눈금 설정
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary) #이미지들을 흑백(cmap=plt.cm.binary)으로 표시.
#     plt.xlabel(class_names[train_labels[i]])        #x축 라벨을 설정
# plt.show()

######################
###     모델구성    ###
######################

model = tf.keras.Sequential([                       ##  간단한 순차적인 구조를 가진 모델을 구성(하나의 입력 하나의 출력)
    tf.keras.layers.Flatten(input_shape=(28,28)),   ##  데이터 형태 변경하지 않고 28*28 1D 벡터로 평탄화(가중치 없음)
    tf.keras.layers.Dense(128, activation='relu'),  ##  뉴런 구성(fully connected layer). 128개 노드.
    tf.keras.layers.Dense(10)                       ##  출력층(softmax층).10개노드. -> 10개 클래스 중 하나에 속할 확률 출력.
])

model.compile(optimizer='adam',         #모델이 인식하는 데이터와 해당 손실 함수를 기반으로 모델 업데이트.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                #   훈련 중 모델이 얼마나 정확한지 측정
              metrics=['accuracy'])      #모니터링

model.fit(train_images, train_labels, epochs=10)        #모델 훈련

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest Accuracy:', test_acc)
##  테스트 세트의 정확도가 훈련 세트의 정확도 보다 낮음 -> 과대적합

####################
###   예측하기    ###
####################

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)  #예측하기
# print(predictions[0])               #0번 이미지 예측값 신뢰도 확인
# print(np.argmax(predictions[0]))    #9번 앵클부츠라고 확신
# print(test_labels[0])               #실제로 뭔지 확인

##  모든 클래스에 대한 예측값 보기
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, predictions[i], test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

########################
###   모델 사용하기   ###
########################

img = test_images[1]
# print(img.shape)      #(28,28) 픽셀의 이미지

img = (np.expand_dims(img,0)) #이미지를 2차원 배열로 만듬
# print(img.shape)

predictions_single = probability_model.predict(img)
# print(predictions_single) #예측

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
np.argmax(predictions_single[0])