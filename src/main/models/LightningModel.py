import lightning as L
from torch import nn, optim


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model,
        lr,
        weight_decay,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _common_step(self, batch):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def training_step(self, batch):
        # print('TRAINING')
        loss, scores, y = self._common_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        # print('VALIDATING')
        loss, scores, y = self._common_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch):
        # print('TESTING')
        loss, scores, y = self._common_step(batch)
        self.log("test_loss", loss)
        return loss
