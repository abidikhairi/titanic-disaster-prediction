import torch
import pandas as pd
from torch import nn
from torch.optim import SGD
from sklearn.model_selection import train_test_split


def tanh(x):
    return torch.sinh(x) / torch.cosh(x)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class MLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()

        self.lin1 = nn.Linear(in_features=n_features, out_features=16, bias=True)
        self.added = nn.Linear(in_features=16, out_features=16, bias=True)
        self.lin2 = nn.Linear(in_features=16, out_features=1, bias=True)

    def forward(self, x):

        z1 = self.lin1(x)

        # a1 = torch.relu(z1)
        a1 = torch.tanh(z1)

        w = torch.relu(self.added(a1))

        z2 = self.lin2(w)
        a2 = torch.sigmoid(z2)
        
        return a2


def main():
    train_file = "data/train-v1.csv"
    data = pd.read_csv(train_file)

    print("Data shape", data.shape)

    x_data = data[['Pclass','Sex_Code','Age','SibSp','Parch','Fare','Embarked_Code']]
    y_data = data['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    x_train = torch.tensor(x_train.values, dtype=torch.float32, requires_grad=True)
    x_test = torch.tensor(x_test.values, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_train.values, dtype=torch.float32, requires_grad=True)
    y_test = torch.tensor(y_test.values, dtype=torch.float32, requires_grad=True)

    model = MLP(n_features=7)
    optimizer = SGD(model.parameters(), lr=0.03)

    loss_fn = nn.BCEWithLogitsLoss()

    model.train()

    for e in range(25):
        # PyTorch
        optimizer.zero_grad()

        y_pred = model(x_train).view(-1)

        loss = loss_fn(y_pred, y_train)

        loss = loss + 1e-9

        loss.backward()
        optimizer.step()

        print(f"Epoch: {e + 1}\tLoss: {loss.item()}")


if __name__ == "__main__":
    main()
