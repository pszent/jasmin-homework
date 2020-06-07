import torch
from torchvision import models


class Resnet50emb(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # output is Bx2048x7x7 here
        x = self.model.avgpool(x)
        # flatten the results to 1dim
        embedding = torch.flatten(x, 1)
        # forward layer gets only Bx2048 params
        x = self.model.fc(embedding)

        return x, embedding
