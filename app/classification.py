import torch
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from app.dataset import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset('../data/')
dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=False, num_workers=4)

model = models.resnet50(pretrained=True)
if torch.cuda.is_available():
    model.cuda()

preds = []
for batch in tqdm(dataloader):
    batch = batch.to(device, dtype=torch.float)
    with torch.no_grad():
        output = model(batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    print(torch.nn.functional.softmax(output[0], dim=0))
