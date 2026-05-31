import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data[:, :3] = resnet.conv1.weight.data
        self.conv1.weight.data[:, 3] = resnet.conv1.weight.data[:, 0]
        
        self.features = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class TransformerBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(4, 64, kernel_size=8, stride=8)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 64))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(64, 768)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        cls_output = x[:, 0]
        return self.fc(cls_output)

class MLPBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x

class RGBBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 512)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MorphologicalBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 128)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GraphBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiModalFusionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn = CNNBranch()
        self.transformer = TransformerBranch()
        self.mlp = MLPBranch()
        self.rgb_cnn = RGBBranch()
        self.morph_branch = MorphologicalBranch()
        self.graph_branch = GraphBranch()

        self.fusion_fc1 = nn.Linear(3648, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, spatial, spectral, rgb, morph, graph):
        cnn_feat = self.cnn(spatial)
        trans_feat = self.transformer(spatial)
        spec_feat = self.mlp(spectral)
        rgb_feat = self.rgb_cnn(rgb)
        morph_feat = self.morph_branch(morph)
        graph_feat = self.graph_branch(graph)

        fused = torch.cat([cnn_feat, trans_feat, spec_feat, rgb_feat, morph_feat, graph_feat], dim=1)

        x = self.relu(self.fusion_fc1(fused))
        x = self.dropout(x)
        output = self.classifier(x)

        return output
