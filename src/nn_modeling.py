import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

class SimpleNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[64, 32, 16]):
        super().__init__()

        #normalize input
        self.input_norm = nn.BatchNorm1d(input_dim)

        #hidden layers
        self.fc = nn.ModuleList()
        for idx, hs in enumerate(hidden_sizes):
            if idx==0:                
                self.fc.append(nn.Linear(input_dim, hs))
            else:
                self.fc.append(nn.Linear(hidden_sizes[idx-1], hs))

        #regression output
        self.out = nn.Linear(hidden_sizes[-1], 1)

        #activation function
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.input_norm(x)

        for layer in self.fc:
            x = self.relu(layer(x))

        return self.out(x)

class TorchRegressorWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, model_kwargs=None, lr=0.001, epochs=10, batch_size=32, device=None):
        """
        model_class: class of the PyTorch model (e.g., SimpleNNRegressor)
        model_kwargs: dict of kwargs to init the model
        lr: learning rate
        epochs: number of training epochs
        batch_size: batch size
        device: 'cuda' or 'cpu'
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = self.model_class(**self.model_kwargs).to(self.device)

        # Store loss history
        self.train_losses = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy(dtype=np.float32)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        #rese history
        self.train_losses = []

        for epoch in range(self.epochs):
            batch_losses = []

            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            
            # Average training loss for this epoch
            epoch_train_loss = np.mean(batch_losses)
            self.train_losses.append(epoch_train_loss)

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
        return out.cpu().numpy().ravel()
    
    def get_parameters(self):
        return self.model.parameters()

    # --- Custom save/load ---
    def __getstate__(self):
        """Save state for joblib"""
        state = self.__dict__.copy()
        # Save model weights instead of raw object
        state["model_state"] = self.model.state_dict()
        state["model"] = None  # don’t pickle torch module directly
        return state

    def __setstate__(self, state):
        """Reload model when joblib.load() is called"""
        self.__dict__.update(state)
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        self.model.load_state_dict(self.__dict__["model_state"])