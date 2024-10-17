import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        loss = torch.mean(self(batch))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

def test_distributed_training():
    model = SimpleModel()
    trainer = Trainer(gpus=2, strategy="ddp", max_epochs=1)
    dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 10))
    train_loader = DataLoader(dataset, batch_size=32)

    trainer.fit(model, train_loader)