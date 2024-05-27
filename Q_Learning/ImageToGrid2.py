from PIL import Image
import numpy as np

def image_to_grid_custom_rgb(image_path, rgb_value1, rgb_value2, grid_size=(961, 500)):
    # 이미지를 RGB 모드로 불러옴
    image = Image.open(image_path).convert('RGB')
    image = image.resize(grid_size)  # 이미지를 지정된 크기로 리사이즈
    
    # 이미지 데이터를 numpy 배열로 변환
    image_array = np.array(image)
    
    # RGB 값과의 거리를 계산하는 함수
    def color_distance(rgb1, rgb2):
        return np.sqrt(np.sum((rgb1 - rgb2) ** 2))
    
    # 그리드 초기화
    grid = np.zeros(grid_size, dtype=int)
    
    # 각 픽셀을 순회하며, 두 RGB 값 중 어느 것과 더 가까운지 판단하여 그리드 설정
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            pixel = image_array[i, j]
            distance_to_rgb_value1 = color_distance(pixel, rgb_value1)
            distance_to_rgb_value2 = color_distance(pixel, rgb_value2)
            
            # 더 가까운 색상에 따라 그리드 값을 설정
            if distance_to_rgb_value1 < distance_to_rgb_value2:
                grid[i, j] = 0
            else:
                grid[i, j] = 1
    
    return grid

# 사용 예시
rgb_value1 = np.array([255, 255, 255])
rgb_value2 = np.array([103, 206, 237])
grid = image_to_grid_custom_rgb('map.jpg', rgb_value1, rgb_value2)
print(grid)

# 그리드를 메모장 파일로 저장
output_file_path = 'map.txt'
np.savetxt(output_file_path, grid, fmt='%d')

print(f"그리드가 {output_file_path} 파일에 저장되었습니다.")
