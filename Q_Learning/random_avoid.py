import random

random_xy=[]
random_cnt=50
grid_size_x = 20
grid_size_y = 20

while len(random_xy) < random_cnt:
    x = random.randint(0, grid_size_x-1)
    y = random.randint(0, grid_size_y-1)
    
    if ((x, y) not in random_xy) or ((x,y) != (0,0)) or ((x,y) != (grid_size_x,grid_size_y)):
        random_xy.append((x, y))

print(random_xy)