from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def image_to_grid(image_path, grid_size=(960, 500)):
    # 이미지 읽어오기
    image = Image.open(image_path).convert('L')  # 이미지를 흑백으로 변환
    image = image.resize(grid_size)  # 이미지를 100x100 크기로 리사이즈
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')  # 축을 표시하지 않음
    # plt.show()
    
    # 이미지 데이터를 numpy 배열로 변환
    image_array = np.array(image)
    
    # 그리드 생성: 검은색이면 1, 하얀색이면 0
    threshold = 200  # 중간값 기준으로 흑백 구분
    grid = (image_array < threshold).astype(int)
    
    return grid

# 사용 예시
grid = image_to_grid('map.jpg')
print(grid)

# 그리드를 메모장 파일로 저장
output_file_path = 'map.txt'
np.savetxt(output_file_path, grid, fmt='%d')

print(f"그리드가 {output_file_path} 파일에 저장되었습니다.")
