import random

random_xy=[]
random_cnt=50
while len(random_xy) < random_cnt:
    x = random.randint(0, 19)
    y = random.randint(0, 19)
    
    if (x, y) not in random_xy:
        random_xy.append((x, y))

print(random_xy)