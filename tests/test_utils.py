import pytest
import numpy as np
import torch
from src.utils import gaussian_radius, draw_umich_gaussian, modified_focal_loss

def test_gaussian_radius():
    det_size = (50, 50) # height, width
    radius = gaussian_radius(det_size, min_overlap=0.7)
    assert isinstance(radius, float)
    assert radius > 0

def test_draw_umich_gaussian():
    # 100x100のゼロ行列ヒートマップ
    heatmap = np.zeros((100, 100), dtype=np.float32)
    center = (50, 50)
    radius = 5
    
    heatmap = draw_umich_gaussian(heatmap, center, radius)
    
    # ピークがcenterにあり、値が1であることを確認
    assert heatmap[50, 50] == 1.0
    # 周辺に値が広がっていることを確認
    assert heatmap[50, 51] > 0
    assert heatmap[50, 51] < 1.0

def test_modified_focal_loss():
    # (B, C, H, W) = (2, 1, 10, 10)
    pred = torch.full((2, 1, 10, 10), 0.1) # 予測値
    target = torch.zeros((2, 1, 10, 10)) # 正解
    
    # ピークを1箇所作成
    target[0, 0, 5, 5] = 1.0
    pred[0, 0, 5, 5] = 0.9 # ピーク付近の高い予測値
    
    loss = modified_focal_loss(pred, target)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
