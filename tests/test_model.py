import pytest
import torch
from src.model import CenterNet

def test_centernet_forward():
    model = CenterNet(num_classes=1)
    # バッチサイズ2, 3チャンネル, 512x512画像のダミーテンソル
    x = torch.randn(2, 3, 512, 512)
    
    out = model(x)
    
    assert 'hm' in out
    assert 'wh' in out
    assert 'offset' in out
    
    # 1/4スケールになっているか確認 (512 / 4 = 128)
    assert out['hm'].shape == (2, 1, 128, 128)
    assert out['wh'].shape == (2, 2, 128, 128)
    assert out['offset'].shape == (2, 2, 128, 128)
    
    # Heatmapの値が0~1の範囲に収まっているか (Sigmoid適用済みのため)
    assert out['hm'].min() >= 0.0
    assert out['hm'].max() <= 1.0
