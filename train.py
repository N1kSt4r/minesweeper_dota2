import os
import cv2
import time
import torch
import numpy as np

# . - closed
# c - clock
# m - mana
# g - junk / endgame
#   - open
# * - mine
symbols = ".cmg *123456"
translate_dict = {char: i for i, char in enumerate(symbols)}

def load_img(path):
    rgb_img = cv2.imread(path)[..., ::-1].copy()
    return torch.as_tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255

def load(dir, name):
    name = dir + name
    print('process', name)
    img = load_img(f'{name}.png')
    with open(f'{name}.txt') as file:
        label = [[translate_dict[char] for char in line.strip('\n')] for line in file if line.strip('\n')]
    label = torch.as_tensor(label, dtype=torch.int64).unsqueeze(0)
    return img,label

def get_unique_names(dir):
    return {name.split('.')[0] for name in os.listdir(dir)}

train_dataset = [load('train/', name) for name in get_unique_names('train')]
val_dataset = [load('val/', name) for name in get_unique_names('val')]

model = torch.nn.Conv2d(3, len(translate_dict), bias=False, stride=36, kernel_size=36)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss()

for i in range(0, 5000):
    loss = 0
    optimizer.zero_grad()
    for image, labels in train_dataset:
        logits = model(image)
        loss = loss + criterion(logits, labels) / torch.prod(torch.as_tensor(labels.shape))
    loss /= len(train_dataset) / 1000
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print('train loss:', loss.item())
        loss = 0
        model.eval()
        for image, labels in val_dataset:
            with torch.no_grad():
                logits = model(image)
                loss = loss + criterion(logits, labels) / torch.prod(torch.as_tensor(labels.shape))
        model.train()
        print('val loss:', loss.item() / len(val_dataset) * 1000)

frame = load_img('val/1555.png')

model.eval()
predict = torch.argmax(model(frame)[0], 0).tolist()

for row in predict:
    for idx in row:
        print(symbols[idx], end='')
    print()

torch.save(model, 'recognizer.pth')