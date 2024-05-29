import cv2
import numpy as np

# 이미지를 불러옵니다.
img = cv2.imread('road_map.jpg')

# 원하는 이미지 크기를 설정합니다. 예: 너비 100, 높이 50
width, height = 600, 600

# 이미지를 원하는 크기로 조정합니다.
resized_img = cv2.resize(img, (width, height))


# 조정된 이미지를 파일로 저장하고 싶다면,
cv2.imwrite('resized_road_map.jpg', resized_img)


# 이미지를 불러옵니다.
img = cv2.imread('resized_road_map.jpg')

# 이미지의 색상을 확인할 범위를 정합니다. 
# OpenCV에서 색상은 BGR 순서를 따릅니다.
colors_to_keep=[]
for i in range(0,30):
    colors_to_keep.append((200+i,200+i,200+i))
colors_to_keep.append((174,174,174))
print(colors_to_keep)
# 결과 이미지를 저장할 배열을 생성합니다. 초기 상태에서는 모든 값을 0으로 설정하여 검은색으로 만듭니다.
result_img = np.zeros_like(img)

# 지정된 색상의 픽셀만 결과 이미지에 복사합니다.
for color in colors_to_keep:
    mask = cv2.inRange(img, color, color)
    result_img[mask == 255] = img[mask == 255]

# 결과 이미지를 보여줍니다.
cv2.imshow('Result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 이미지를 파일로 저장하고 싶다면,
# cv2.imwrite('result_image_path.jpg', result_img)
