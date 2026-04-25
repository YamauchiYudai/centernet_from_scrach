import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset import WheatCenterNetDataset, WheatInferenceDataset

class WheatDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, image_dir, batch_size=8, input_size=512, num_workers=4):
        """
        Global Wheat Detection 用の DataModule
        csv_path: train.csv のパス
        image_dir: 画像ディレクトリのパス
        ここはバッチサイズに合わせて調整してください (batch_size)
        """
        super().__init__()
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        # メタデータの読み込み
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            # ダミーデータ（TDDやテスト用）
            print(f"Warning: {self.csv_path} not found. Using dummy data.")
            df = pd.DataFrame({
                'image_id': ['dummy1', 'dummy2'],
                'bbox': ['[10, 10, 50, 50]', '[20, 20, 60, 60]']
            })
            
        # 画像IDベースでTrain/Valを分割 (Hold-out検証)
        image_ids = df['image_id'].unique()
        train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
        
        train_df = df[df['image_id'].isin(train_ids)]
        val_df = df[df['image_id'].isin(val_ids)]
        
        # Datasetのインスタンス化
        self.train_dataset = WheatCenterNetDataset(
            df=train_df, 
            image_dir=self.image_dir,
            input_size=self.input_size
        )
        
        self.val_dataset = WheatCenterNetDataset(
            df=val_df, 
            image_dir=self.image_dir,
            input_size=self.input_size
        )
        
        # 推論用Dataset (ここでは検証用データを使ってテスト推論とする)
        self.predict_dataset = WheatInferenceDataset(
            df=val_df,
            image_dir=self.image_dir,
            input_size=self.input_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
