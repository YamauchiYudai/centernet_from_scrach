import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.model import CenterNet
from src.utils import modified_focal_loss

def decode_centernet(hm, wh, offset, K=100, down_ratio=4):
    """
    ヒートマップからピークを抽出し、バウンディングボックスをデコードする。
    hm: (B, C, H, W)
    wh: (B, 2, H, W)
    offset: (B, 2, H, W)
    """
    batch_size, num_classes, height, width = hm.shape
    
    # ローカルマキシマムの取得 (3x3 MaxPool)
    hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hmax == hm).float()
    hm = hm * keep
    
    # FlattenしてTop-Kを取得
    hm = hm.view(batch_size, -1)
    scores, inds = torch.topk(hm, K, dim=1)
    
    # クラスと座標のインデックスを計算
    classes = (inds / (height * width)).int()
    inds = inds % (height * width)
    ys = (inds / width).int().float()
    xs = (inds % width).int().float()
    
    # 抽出したインデックスに該当するwhとoffsetを取得
    # wh, offset: (B, 2, H, W) -> Flatten -> 抽出
    inds = inds.unsqueeze(1).expand(batch_size, 2, K).long() # (B, 2, K)
    
    wh = wh.view(batch_size, 2, -1)
    wh = torch.gather(wh, 2, inds) # (B, 2, K)
    
    offset = offset.view(batch_size, 2, -1)
    offset = torch.gather(offset, 2, inds) # (B, 2, K)
    
    # 中心座標にオフセットを足す
    xs = xs + offset[:, 0, :]
    ys = ys + offset[:, 1, :]
    
    # Bounding Boxの計算 (出力マップスケールでの [xmin, ymin, width, height])
    half_w = wh[:, 0, :] / 2
    half_h = wh[:, 1, :] / 2
    
    bboxes = torch.stack([
        xs - half_w,
        ys - half_h,
        wh[:, 0, :],
        wh[:, 1, :]
    ], dim=2) # (B, K, 4)
    
    # オリジナルの入力画像スケールに戻す
    bboxes = bboxes * down_ratio
    
    # 最終的な予測 [xmin, ymin, width, height, score, class]
    detections = torch.cat([
        bboxes, 
        scores.unsqueeze(2), 
        classes.unsqueeze(2).float()
    ], dim=2) # (B, K, 6)
    
    return detections

class CenterNetLightningModule(pl.LightningModule):
    def __init__(self, num_classes=1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = CenterNet(num_classes=num_classes)
        self.lr = lr
        
    def forward(self, x):
        return self.model(x)
        
    def _compute_loss(self, preds, batch):
        # 予測
        pred_hm = preds['hm']
        pred_wh = preds['wh']
        pred_offset = preds['offset']
        
        # 正解
        target_hm = batch['hm']
        target_wh = batch['wh']
        target_offset = batch['offset']
        reg_mask = batch['reg_mask']
        
        # Heatmap Loss (Modified Focal Loss)
        hm_loss = modified_focal_loss(pred_hm, target_hm)
        
        # WH Loss (L1 Loss, マスクされた領域のみ)
        # expand mask for 2 channels
        mask_2d = reg_mask.expand_as(pred_wh)
        wh_loss = F.l1_loss(pred_wh * mask_2d, target_wh * mask_2d, reduction='sum') / (mask_2d.sum() + 1e-4)
        
        # Offset Loss (L1 Loss)
        offset_loss = F.l1_loss(pred_offset * mask_2d, target_offset * mask_2d, reduction='sum') / (mask_2d.sum() + 1e-4)
        
        # Total Loss (重み付けは論文を参照：hm:1.0, wh:0.1, off:1.0)
        total_loss = hm_loss + 0.1 * wh_loss + 1.0 * offset_loss
        
        return total_loss, hm_loss, wh_loss, offset_loss
        
    def training_step(self, batch, batch_idx):
        preds = self(batch['image'])
        loss, hm_loss, wh_loss, off_loss = self._compute_loss(preds, batch)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_hm_loss', hm_loss)
        self.log('train_wh_loss', wh_loss)
        self.log('train_off_loss', off_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch['image'])
        loss, hm_loss, wh_loss, off_loss = self._compute_loss(preds, batch)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_hm_loss', hm_loss)
        return loss
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 推論用データローダーは画像とimage_idのみを返す想定
        images = batch['image']
        image_ids = batch['image_id']
        
        preds = self(images)
        
        # デコード処理 (B, K, 6) -> [xmin, ymin, w, h, score, class]
        detections = decode_centernet(
            preds['hm'], preds['wh'], preds['offset'], K=100, down_ratio=4
        )
        
        return {
            'image_ids': image_ids,
            'detections': detections
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # 必要に応じてスケジューラを追加
        return optimizer
