import json

import torch
from torch.utils.data import DataLoader
from torchvision import models

from app.dataset import CustomDataset

with open('class_indexes.json', 'r') as f:
    labels = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset('../data/')
dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=False, num_workers=4)

model = models.resnet50(pretrained=True)
if torch.cuda.is_available():
    model.cuda()

for batch in dataloader:
    batch = batch.to(device, dtype=torch.float)
    with torch.no_grad():
        output = model(batch)

    softmaxed = torch.softmax(output, dim=1)

    # get the label indexes
    label_indexes = torch.argmax(output, dim=1).cpu().numpy()

    # get means and variances - could be done with the torch.var_mean() too but this is more readable
    mean = torch.mean(output, dim=1).cpu().numpy()
    variance = torch.var(output, dim=1).cpu().numpy()

    for index, value in enumerate(mean):
        print(f'mean: {value} variance: {variance[index]} of image #{index + 1}')
    print('----------')

    # get the 2 most probable prediction
    top2 = torch.topk(softmaxed, 2, dim=1)
    probabilities = top2[0].cpu().numpy()
    indices = top2[1].cpu().numpy()

    for index, value in enumerate(indices):
        print(
            f'this is {probabilities[index][0] * 100:.2f}% a {labels[str(value[0])]} and '
            f'{probabilities[index][1] * 100:.2f}% a {labels[str(value[1])]}')
