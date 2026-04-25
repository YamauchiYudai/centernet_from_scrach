import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import ast
import math
from src.utils import gaussian_radius, draw_umich_gaussian

class WheatCenterNetDataset(Dataset):
    def __init__(self, df, image_dir, input_size=512, down_ratio=4, transforms=None):
        """
        学習用のDataset
        df: train.csv のデータ (image_id, bbox 等が含まれる)
        image_dir: 画像ディレクトリのパス
        input_size: モデルへの入力画像サイズ（正方形を想定）
        down_ratio: CenterNetの出力ストライド（通常は4）
        transforms: Albumentations等のData Augmentation（プレースホルダー）
        """
        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = image_dir
        self.input_size = input_size
        self.down_ratio = down_ratio
        self.output_size = input_size // down_ratio
        self.transforms = transforms
        
        # クラス数=1 (小麦の穂)
        self.num_classes = 1
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = f"{self.image_dir}/{image_id}.jpg"
        
        # 画像読み込み
        img = cv2.imread(image_path)
        if img is None:
            # 万が一読み込めなかった場合のダミー画像（テスト等用）
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        records = self.df[self.df['image_id'] == image_id]
        
        # バウンディングボックスの取得
        # bboxは '[xmin, ymin, width, height]' の文字列またはリスト形式と想定
        boxes = []
        for bbox_str in records['bbox'].values:
            if isinstance(bbox_str, str):
                bbox = ast.literal_eval(bbox_str)
            else:
                bbox = bbox_str
            boxes.append(bbox)
            
        # --- ここに Albumentations などのData Augmentationを入れる場所です ---
        if self.transforms:
            # img = self.transforms(image=img, bboxes=boxes)['image']
            pass
        else:
            # Augmentationがない場合はリサイズのみ（bboxのスケール変換も本来は必要）
            img = cv2.resize(img, (self.input_size, self.input_size))
            # 正規化
            img = img.astype(np.float32) / 255.0
            
        # ターゲットテンソルの初期化
        hm = np.zeros((self.num_classes, self.output_size, self.output_size), dtype=np.float32)
        wh = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        offset = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg_mask = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        
        # （簡易的な処理）オリジナル画像のサイズを1024x1024と仮定し、bboxをスケール変換
        # ※ 実務ではtransforms(Albumentations)でbboxも変換させるのがベストプラクティスです。
        scale_x = self.input_size / 1024.0
        scale_y = self.input_size / 1024.0
        
        for bbox in boxes:
            xmin, ymin, w, h = bbox
            xmin, w = xmin * scale_x, w * scale_x
            ymin, h = ymin * scale_y, h * scale_y
            xmax, ymax = xmin + w, ymin + h
            
            # 出力マップ上の座標に変換
            xmin_out, ymin_out = xmin / self.down_ratio, ymin / self.down_ratio
            xmax_out, ymax_out = xmax / self.down_ratio, ymax / self.down_ratio
            w_out, h_out = w / self.down_ratio, h / self.down_ratio
            
            # 中心座標
            ctx = xmin_out + w_out / 2.0
            cty = ymin_out + h_out / 2.0
            
            # 整数座標
            ctx_int = int(ctx)
            cty_int = int(cty)
            
            # 範囲外チェック
            if ctx_int < 0 or ctx_int >= self.output_size or cty_int < 0 or cty_int >= self.output_size:
                continue
                
            # ガウス分布の半径計算と描画 (クラスインデックス=0固定)
            radius = gaussian_radius((math.ceil(h_out), math.ceil(w_out)))
            radius = max(0, int(radius))
            hm[0] = draw_umich_gaussian(hm[0], (ctx_int, cty_int), radius)
            
            # whとoffsetの格納
            wh[0, cty_int, ctx_int] = w_out
            wh[1, cty_int, ctx_int] = h_out
            
            offset[0, cty_int, ctx_int] = ctx - ctx_int
            offset[1, cty_int, ctx_int] = cty - cty_int
            
            # maskの格納
            reg_mask[0, cty_int, ctx_int] = 1.0
            
        # PyTorchテンソルへの変換 (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        hm = torch.from_numpy(hm)
        wh = torch.from_numpy(wh)
        offset = torch.from_numpy(offset)
        reg_mask = torch.from_numpy(reg_mask)
        
        return {
            'image': img,
            'hm': hm,
            'wh': wh,
            'offset': offset,
            'reg_mask': reg_mask
        }

class WheatInferenceDataset(Dataset):
    def __init__(self, df, image_dir, input_size=512, transforms=None):
        """
        推論用のDataset (画像とimage_idのみを返す)
        """
        self.image_ids = df['image_id'].unique()
        self.image_dir = image_dir
        self.input_size = input_size
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = f"{self.image_dir}/{image_id}.jpg"
        
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transforms:
            pass
        else:
            img = cv2.resize(img, (self.input_size, self.input_size))
            img = img.astype(np.float32) / 255.0
            
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return {
            'image': img,
            'image_id': image_id
        }
