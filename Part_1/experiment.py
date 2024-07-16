import torch
from torch.utils.data import DataLoader, TensorDataset

PADDING = 0
batch_size = 32


def get_char_to_idx():
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd']
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    return char_to_idx


def get_X_y():
    char_to_idx = get_char_to_idx()
    with open('pos_examples', 'r') as pos_file:
        pos_examples = pos_file.readlines()
    with open('neg_examples', 'r') as neg_file:
        neg_examples = neg_file.readlines()
    examples = pos_examples + neg_examples
    labels = [1] * len(pos_examples) + [0] * len(neg_examples)
    examples = [example.strip() for example in examples]
    examples = [[char_to_idx[char] for char in example] for example in examples]
    max_len_example = len(max(examples, key=len))
    examples = [[PADDING] * (max_len_example - len(example)) + example for example in examples]
    X = torch.tensor(examples)
    y = torch.tensor(labels)
    return X, y


def split_dataset(data, train_ratio, test_ratio):
    train_size = int(train_ratio * data.shape[0])
    test_size = int(test_ratio * data.shape[0])
    return data[:train_size], data[train_size:train_size + test_size]


class LSTM(torch.nn.Module):
    def __init__(self, char_embed_size, num_char_embedding=14, hidden_size1=100, hidden_size2=50, num_layers=1,
                 batch_size=32):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size1
        self.num_layers = num_layers
        self.char_embedding = torch.nn.Embedding(num_char_embedding, char_embed_size)
        self.lstm = torch.nn.LSTM(char_embed_size, hidden_size1, num_layers,
                                  batch_first=True)
        self.fc = torch.nn.Linear(hidden_size1, hidden_size2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size2, 1)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.char_embedding(x)
        x = self.lstm(x)[0][:, -1, :]
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LstmTrainer:

    def train(
            self, X: torch.Tensor, y: torch.Tensor
    ) -> LSTM:
        X_train, X_val = split_dataset(X, 0.8, 0.2)
        y_train, y_val = split_dataset(y, 0.8, 0.2)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True
        )

        self.model = LSTM(10)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    val_loss += criterion(y_pred, y_batch.unsqueeze(1).float())

                val_loss /= len(val_loader)
                print(f'Epoch: {epoch}, Loss: {val_loss}')

        return self.model


if __name__ == "__main__":
    X, y = get_X_y()
    X_train, X_test = split_dataset(X, 0.8, 0.2)
    y_train, y_test = split_dataset(y, 0.8, 0.2)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True
    )
