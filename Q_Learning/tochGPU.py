# GPU SETTING
# version : python=3.9, cudatoolkit=11.2, cudnn=8.1.0, tensor=2.11
# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
print(torch.cuda.is_available())