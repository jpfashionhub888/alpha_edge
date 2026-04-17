# models/lstm_model.py

"""
LSTM Neural Network for time series prediction.
Sees patterns across time, not just individual features.
This is the 4th model in our ensemble.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """PyTorch LSTM model architecture."""

    def __init__(self, input_size, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class LSTMPredictor:
    """
    LSTM wrapper that matches our TechnicalPredictor
    interface so it plugs into the ensemble seamlessly.
    """

    def __init__(self, sequence_length=20,
                 hidden_size=64, num_layers=2,
                 epochs=50, learning_rate=0.001,
                 batch_size=32):

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size

        self.model = None
        self.feature_names = []
        self.trained = False

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def _create_sequences(self, X, y=None):
        """Convert flat data into sequences for LSTM."""

        data = X.values if hasattr(X, 'values') else X
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)

            if y is not None:
                y_vals = (
                    y.values if hasattr(y, 'values') else y
                )
                targets.append(y_vals[i + self.sequence_length])

        sequences = np.array(sequences, dtype=np.float32)

        # Replace NaN and Inf
        sequences = np.nan_to_num(
            sequences, nan=0.0, posinf=1.0, neginf=-1.0
        )

        if y is not None:
            targets = np.array(targets, dtype=np.float32)
            return sequences, targets
        else:
            return sequences

    def _normalize(self, data):
        """Simple normalization to prevent exploding gradients."""

        mean = np.nanmean(data, axis=(0, 1), keepdims=True)
        std = np.nanstd(data, axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1, std)

        normalized = (data - mean) / std
        normalized = np.nan_to_num(
            normalized, nan=0.0, posinf=1.0, neginf=-1.0
        )

        self._mean = mean
        self._std = std

        return normalized

    def _normalize_predict(self, data):
        """Normalize using stored mean/std."""

        if hasattr(self, '_mean') and hasattr(self, '_std'):
            normalized = (data - self._mean) / self._std
            normalized = np.nan_to_num(
                normalized, nan=0.0, posinf=1.0, neginf=-1.0
            )
            return normalized
        return data

    def train(self, X, y):
        """Train the LSTM model."""

        self.feature_names = list(X.columns)

        if len(X) < self.sequence_length + 20:
            logger.warning(
                "Not enough data for LSTM, skipping"
            )
            self.trained = False
            return self

        # Create sequences
        sequences, targets = self._create_sequences(X, y)

        if len(sequences) < 20:
            self.trained = False
            return self

        # Normalize
        sequences = self._normalize(sequences)

        # Convert to tensors
        X_tensor = torch.FloatTensor(sequences).to(self.device)
        y_tensor = torch.FloatTensor(targets).to(self.device)
        y_tensor = y_tensor.unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Create model
        input_size = sequences.shape[2]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )

        # Train
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                optimizer.step()
                total_loss += loss.item()
                batches += 1

        self.trained = True
        avg_loss = total_loss / max(batches, 1)
        logger.info(
            f"LSTM training complete."
            f" Final loss: {avg_loss:.4f}"
        )

        return self

    def predict(self, X):
        """Generate predictions."""

        if not self.trained or self.model is None:
            return np.full(len(X), 0.5)

        self.model.eval()

        with torch.no_grad():
            if len(X) >= self.sequence_length:
                sequences = self._create_sequences(X)
                sequences = self._normalize_predict(sequences)

                X_tensor = torch.FloatTensor(
                    sequences
                ).to(self.device)

                outputs = self.model(X_tensor)
                preds = outputs.cpu().numpy().flatten()

                # Pad the beginning with 0.5
                full_preds = np.full(len(X), 0.5)
                full_preds[-len(preds):] = preds

                return full_preds
            else:
                return np.full(len(X), 0.5)

    def predict_proba(self, X):
        """
        Return probabilities in sklearn format.
        Required for ensemble compatibility.
        """

        preds = self.predict(X)

        # Return as 2D array [prob_class_0, prob_class_1]
        proba = np.column_stack([1 - preds, preds])
        return proba