import torch
import torch.nn as nn

class Hoda_SadigAli(nn.Module):
    def __init__(self,):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Dropout(0.1)         
        )
        #torch.Size([x, 64, 18, 18])
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Dropout(0.2)
        )
        #torch.Size([x, 128, 7, 7])
        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Dropout(0.3)
        )
        #torch.Size([x, 256, 3, 3])
        self.block_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Dropout(0.4)

        )
        #torch.Size([x, 512, 1, 1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 ,out_features=1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.Linear(in_features=1024,out_features=10))
        #torch.Size([x, 10])
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.block_3(x)
        # print(x.shape)
        x = self.block_4(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# model= Hoda_SadigAli().to(device)
# model_2.forward(torch.randn(2, 1, 40, 40))