import json
import os
import time
import torch
from torch.utils.data import DataLoader
from app.resnet_embedding import Resnet50emb

from app.dataset import CustomDataset

with open(os.path.join(os.getcwd(), 'class_indexes.json'), 'r') as f:
    labels = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset('../data/')
dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=False, num_workers=4)

model = Resnet50emb()
if torch.cuda.is_available():
    model.cuda()

# switch to eval so inference is quicker
model.eval()

start = time.time()

for batch in dataloader:
    batch = batch.to(device, dtype=torch.float)
    with torch.no_grad():
        embedding, output = model(batch)

    softmaxed = torch.softmax(embedding, dim=1)

    # get means and variances - could be done with the torch.var_mean() too but this is more readable
    mean = torch.mean(embedding, dim=1).cpu().numpy()
    variance = torch.var(embedding, dim=1).cpu().numpy()

    for index, value in enumerate(mean):
        print(f'mean: {value} variance: {variance[index]} of image #{index + 1}')
    print('----------')

    # get the 2 most probable prediction
    top2 = torch.topk(softmaxed, 2, dim=1)
    probabilities = top2[0].cpu().numpy()
    indices = top2[1].cpu().numpy()

    for index, value in enumerate(indices):
        print(
            f'image #{index+1} is {probabilities[index][0] * 100:.2f}% a {labels[str(value[0])]} and '
            f'{probabilities[index][1] * 100:.2f}% a {labels[str(value[1])]}')

    print(f'inference in python took {time.time()-start:.2f} seconds')
