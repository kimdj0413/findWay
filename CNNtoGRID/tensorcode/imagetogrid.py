from PIL import Image
import matplotlib.pylab as plt
import numpy as np
import os
import tensorflow as tf
import pathlib
from tensorflow import keras
import pandas as pd
import shutil
import matplotlib.patches as mpatches
from skimage.util import view_as_windows

#####################################
###     이미지를 그리드로 변경      ###
#####################################

img = Image.open('C:/waytoMap/fruitCNN/fruits/fruits_image16.jpg')
# img = Image.open('C:/waytoMap/strawberries.jpg')
x = 1000
y = 1000
row = 10
reduce_cnt = 1
img_resize = img.resize((x,y))

fig, axs = plt.subplots(row, row, figsize=(10, 20))
cnt=0
os.makedirs('fruits_example', exist_ok=True)
os.chdir('fruits_example')

for i in range(row):
  for j in range(row):
    img_cropped = img_resize.crop((j*x/row,i*y/row,(j+1)*x/row,(i+1)*y/row))
    axs[i, j].imshow(np.array(img_cropped))
    axs[i, j].axis('off')
    if img_cropped.mode == 'RGBA':
        img_cropped = img_cropped.convert('RGB')
    img_cropped.save('fruits'+str(cnt)+".jpg")
    cnt+=1

##############################
###     이미지 별 학습      ###
##############################
result = []
result_score = []
class_names = ['apple', 'banana', 'coconut', 'grape', 'orange']
model = keras.models.load_model('C:/waytoMap/fruitCNN86abcgo.keras')
for i in range(row*row):
    image_path = "C:/waytoMap/fruits_example/fruits"+str(i)+".jpg"
    img = tf.keras.utils.load_img(
        image_path, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    result.append(class_names[np.argmax(score)])
    result_score.append(np.max(score))

##################################
###     그리드 크기 줄이기      ###
#################################

result_array = np.array(result).reshape(row, row)
result_score_array = np.array(result_score).reshape(row, row)
df_result = pd.DataFrame(result_array)
df_score = pd.DataFrame(result_score_array)
if reduce_cnt == 0:
   resized_df = df_result
else: 
    for i in range(0,reduce_cnt):
        filter = view_as_windows(df_score.to_numpy(), (2, 2))
        max_value = np.unravel_index(filter.reshape(filter.shape[:2] + (-1,)).argmax(axis=-1), filter.shape[2:])
        df_score = pd.DataFrame(df_score.to_numpy()[max_value[0] + np.arange(row-i-1)[:, None], max_value[1] + np.arange(row-i-1)])
        df_result = pd.DataFrame(df_result.to_numpy()[max_value[0] + np.arange(row-i-1)[:, None], max_value[1] + np.arange(row-i-1)])
        resized_df= df_result

##############################
###     결과값 그리기       ###
##############################

fruit_color = {
    'apple': 'red',
    'orange': 'orange',
    'coconut': 'brown',
    'banana': 'yellow',
    'grape': 'purple',
}

color_df = resized_df.applymap(fruit_color.get)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow([[plt.cm.colors.to_rgba(c) for c in row] for row in color_df.values])

ax.set_xticks(np.arange(len(resized_df.columns)))
ax.set_yticks(np.arange(len(resized_df)))
ax.set_xticklabels(resized_df.columns)
ax.set_yticklabels(resized_df.index)

patches = [mpatches.Patch(color=color, label=fruit) for fruit, color in fruit_color.items()]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

print(resized_df)
print(df_score)
plt.show()
shutil.rmtree('C:/waytoMap/fruits_example')