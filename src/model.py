import torch
import torch.nn as nn
import torchvision.models as models

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # チェッカーボードノイズ対策のため、Deconvの前に3x3 Convを挟む
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Deconvで2倍にアップサンプリング
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.deconv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class CenterNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Backbone: ResNet18 (AdaptiveAvgPool2dとfcは除外)
        # pretrained=True にしたい場合は weights=models.ResNet18_Weights.DEFAULT を指定する
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1, # stride=1
            resnet.layer2, # stride=2 (1/8)
            resnet.layer3, # stride=2 (1/16)
            resnet.layer4  # stride=2 (1/32) -> in_channels=512
        )
        
        # Neck: 1/32 -> 1/16 -> 1/8 -> 1/4 (解像度を1/4スケールに戻す)
        self.neck = nn.Sequential(
            DeconvLayer(512, 256),
            DeconvLayer(256, 128),
            DeconvLayer(128, 64)
        )
        
        # Head: Heatmap, WH, Offset
        # ヒートマップは Sigmoid を適用する前までを出力（Loss関数内で対応するため、または外部でSigmoidを適用）
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        # Sigmoidのバイアス初期化の工夫 (CenterNet論文参照: -2.19)
        self.heatmap_head[-1].bias.data.fill_(-2.19)
        
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.backbone(x)  # (B, 512, H/32, W/32)
        feat = self.neck(feat)   # (B, 64, H/4, W/4)
        
        hm = self.heatmap_head(feat)
        hm = torch.sigmoid(hm) # CenterNetではヒートマップにSigmoidを適用する
        
        wh = self.wh_head(feat)
        offset = self.offset_head(feat)
        
        return {
            'hm': hm,
            'wh': wh,
            'offset': offset
        }
