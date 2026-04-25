# Global Wheat Detection with CenterNet

このプロジェクトは、Kaggleの「Global Wheat Detection」向けに構築されたCenterNetのPyTorch Lightningベースコードです。

## 特徴
- **CenterNetアーキテクチャ**: ResNet18バックボーン + 3 Deconvレイヤー (チェッカーボードノイズ対策の3x3 Conv含む)
- **Loss関数**: HeatmapにはModified Focal Loss、SizeとOffsetにはL1 Lossを使用
- **PyTorch Lightning**: `CenterNetLightningModule` と `WheatDataModule` によるモジュール化された学習パイプライン
- **TDD (テスト駆動開発) 対応**: `tests/` ディレクトリに各種コンポーネントのテストを配置
- **Docker環境**: 依存関係を隔離し、再現性を確保するための `Dockerfile` と `docker-compose.yml` を提供

## ディレクトリ構成
```text
.
├── src/
│   ├── dataset.py          # Datasetクラス (WheatCenterNetDataset, WheatInferenceDataset)
│   ├── datamodule.py       # LightningDataModule (WheatDataModule)
│   ├── model.py            # CenterNet本体のアーキテクチャ定義
│   ├── lightning_module.py # Loss計算および推論用デコード(decode_centernet)の実装
│   └── utils.py            # ガウス分布生成やFocal Lossの実装
├── tests/                  # 単体テストコード (pytest)
├── docs/                   # タスク状態などのドキュメント
├── train.py                # 学習用エントリーポイントスクリプト
├── Dockerfile              # 開発用Dockerイメージ
├── docker-compose.yml      # Docker環境起動用Composeファイル
└── requirements.txt        # 必要なPythonパッケージ
```

## 使い方

### 1. 環境構築 (Docker)
プロジェクトルートで以下のコマンドを実行してコンテナを起動し、シェルに入ります。
```bash
docker compose run --rm app /bin/bash
```

### 2. データセットの配置
`data/` ディレクトリ（または任意のディレクトリ）に以下のデータを配置してください。
- `train.csv`
- `train/` (学習画像のディレクトリ)

### 3. 学習の実行 (fast_dev_run)
正常に実行できるかどうかの簡易テストを行うには以下を実行します。
```bash
python train.py --fast_dev_run
```

### 4. 本学習
```bash
python train.py --csv_path data/train.csv --image_dir data/train --max_epochs 10 --batch_size 8
```

## カスタマイズについて
コード内のコメントに記載の通り、以下は実務の要件に合わせて適宜追加・調整してください。
- `src/dataset.py`: Albumentations等のData Augmentation処理
- `train.py`: WandB等のロガー設定、ModelCheckpoint等のコールバック設定