
import numpy as np
import torch
from rtdl_revisiting_models import ResNet as rtdlResNet
from sklearn.base import BaseEstimator
from torch import nn, optim
from torch.utils.data import DataLoader

from main.models.base_models.CustomDataset import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("medium")


class ResNet(BaseEstimator):
    """ResNet architecture based on:
    Revisiting Deep Learning Models for Tabular Data
    https://arxiv.org/abs/2106.11959

    hyperparameters based on table 14 of the paper:
    Table 14: ResNet hyperparameter space. Here (A) = {CA, AD, HE, JA, HI, AL} and
    (B) = {EP, YE, CO, YA, MI}
    Parameter (Datasets) Distribution
    # Layers (A) UniformInt[1, 8], (B) UniformInt[1, 16]
    Layer size (A) UniformInt[64, 512], (B) UniformInt[64, 1024]
    Hidden factor (A,B) Uniform[1, 4]
    Hidden dropout (A,B) Uniform[0, 0.5]
    Residual dropout (A,B) {0, Uniform[0, 0.5]}
    Learning rate (A,B) LogUniform[1e-5, 1e-2]
    Weight decay (A,B) {0, LogUniform[1e-6, 1e-3]}
    Category embedding size ({AD}) UniformInt[64, 512]
    # Iterations 100

    Hyperparameter space was chosen based on (A) = {CA, AD, HE, JA, HI, AL}.
    """

    def __init__(
        self,
        *,
        # d_in: int, # we'll figure it out once we receive the input samples
        # d_out: Optional[int], # we'll figure it out once we receive the input samples' labels
        n_blocks: int,
        d_block: int,
        d_hidden: int | None,
        d_hidden_multiplier: float | None,
        dropout1: float,
        dropout2: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        """Args:
        d_in: the input size.
        d_out: the output size.
        n_blocks: the number of blocks.
        d_block: the block width (i.e. its input and output size).
        d_hidden: the block's hidden width.
        d_hidden_multipler: the alternative way to set `d_hidden` as
            `int(d_block * d_hidden_multipler)`.
        dropout1: the hidden dropout rate.
        dropout2: the residual dropout rate.
        """
        # self.d_in = d_in
        # self.d_out = d_out
        self.n_blocks = n_blocks  # # Layers (A) UniformInt[1, 8],
        self.d_block = d_block  # Layer size (A) UniformInt[64, 512],
        self.d_hidden = d_hidden  # Hidden factor (A,B) Uniform[1, 4]
        self.d_hidden_multiplier = d_hidden_multiplier  # will be None here since we'll already be using d_hidden
        self.dropout1 = dropout1  # Hidden dropout (A,B) Uniform[0, 0.5]
        self.dropout2 = dropout2  # Residual dropout (A,B) {0, Uniform[0, 0.5]}
        self.lr = lr  # Learning rate (A,B) LogUniform[1e-5, 1e-2]
        self.weight_decay = weight_decay  # Weight decay (A,B) {0, LogUniform[1e-6, 1e-3]}
        self.precision = "bf16-mixed"
        self.patience = 16
        self.batch_size = 128
        self.num_epochs = 100
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def _init_model(self, input_size, output_size):
        self.model = rtdlResNet(
            d_in=input_size,
            d_out=output_size,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            d_hidden=self.d_hidden,
            d_hidden_multiplier=self.d_hidden_multiplier,
            dropout1=self.dropout1,
            dropout2=self.dropout2,
        ).to(device)

    def fit(self, X, y):
        self._init_model(X.shape[1], len(np.unique(y)))

        custom_train_dataset = CustomDataset(X, y)

        train_loader = DataLoader(
            dataset=custom_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Training loop
        for epoch in range(self.num_epochs):
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
                    )

        self.model.eval()

    def forward(self, X):
        X = self.model(torch.tensor(X, dtype=torch.float32).to(device))
        return self.softmax(X)

    def predict_proba(self, X):
        return self.forward(X).cpu().detach().numpy()

    def predict(self, X):
        prob = self.forward(X)
        return torch.argmax(prob, axis=1).cpu().detach().numpy()


if __name__ == "__main__":
    # NOTE: all code snippets can be copied and executed as-is.
    import torch
    import torch.nn.functional as F
    import torch.optim
    from torch import nn

    batch_size = 2

    # Continuous features.
    n_cont_features = 3
    x_cont = torch.randn(batch_size, n_cont_features)

    # Categorical features.
    cat_cardinalities = [
        4,  # Allowed values: [0, 1, 2, 3].
        7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
    ]
    n_cat_features = len(cat_cardinalities)
    x_cat = torch.column_stack([torch.randint(0, c, (batch_size,)) for c in cat_cardinalities])
    assert x_cat.dtype == torch.int64
    assert x_cat.shape == (batch_size, n_cat_features)

    # MLP-like models (e.g. MLP and ResNet) require
    # categorical features to be encoded as continuous features.
    # One way to achieve that is the one-hot encoding
    # (for features with high cardinality, embeddings can be a better choice).
    x_cat_ohe = [F.one_hot(cat_column, c) for cat_column, c in zip(x_cat.T, cat_cardinalities, strict=False)]
    x = torch.column_stack([x_cont] + x_cat_ohe)
    assert x.shape == (batch_size, n_cont_features + sum(cat_cardinalities))

    d_out = 1  # For example, a single regression task.
    model = rtdlResNet(
        d_in=n_cont_features + sum(cat_cardinalities),
        d_out=d_out,
        n_blocks=2,
        d_block=192,
        d_hidden=None,
        d_hidden_multiplier=2.0,
        dropout1=0.15,
        dropout2=0.0,
    )
    y_pred = model(x)
    assert y_pred.shape == (batch_size, d_out)
