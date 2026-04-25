import pytorch_lightning as pl
from src.datamodule import WheatDataModule
from src.lightning_module import CenterNetLightningModule
import argparse

def main():
    parser = argparse.ArgumentParser(description="Global Wheat Detection with CenterNet")
    parser.add_argument('--csv_path', type=str, default='data/train.csv', help='Path to train.csv')
    parser.add_argument('--image_dir', type=str, default='data/train', help='Path to train images directory')
    parser.add_argument('--max_epochs', type=int, default=10, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a fast dev run for debugging')
    args = parser.parse_args()

    print("Initializing DataModule...")
    datamodule = WheatDataModule(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size
    )

    print("Initializing Model...")
    model = CenterNetLightningModule(num_classes=1, lr=1e-4)

    print("Initializing Trainer...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        accelerator='auto', # GPUがあれば使用
        devices=1,
        # logger=..., # WandB or TensorBoard
        # callbacks=[...] # ModelCheckpoint, EarlyStopping
    )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    
    print("Training finished.")

if __name__ == '__main__':
    main()
