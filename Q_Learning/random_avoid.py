import random

# random_xy=[]
# random_cnt=50
# grid_size_x = 20
# grid_size_y = 20

# while len(random_xy) < random_cnt:
#     x = random.randint(0, grid_size_x-1)
#     y = random.randint(0, grid_size_y-1)
    
#     if ((x, y) not in random_xy) or ((x,y) != (0,0)) or ((x,y) != (grid_size_x-1,grid_size_y-1)):
#         random_xy.append((x, y))

# print(random_xy)
# print(random.randint(0, 3))
# 파일을 쓰기 모드로 열기
with open('output.txt', 'w') as file:
    # 예시로 10번 반복
    for i in range(10):
        # 파일에 문자열 쓰기
        file.write(f'반복되는 줄 {i+1}\n')

# 파일이 자동으로 닫히며 저장됩니다.
