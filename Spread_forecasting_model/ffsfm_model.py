"""
=============================================================================
ffsfm_model.py  —  ConvBiLSTM Architecture
=============================================================================
Input  : (batch, T=14, Z, F=65)
Output : (batch, 7, Z)   — fire probability per zone per day

Architecture
------------
  TimeDistributed Conv1D × 2   → spatial spread patterns across zones
  BiLSTM × 2                   → temporal dynamics (forward + backward)
  Dense decoder                → 7-day × Z spread probability map

One model per district (zone counts differ per district):
  Banke   Z=14
  Bardiya Z=15
  Surkhet Z=19
  Dang    Z=17
  Salyan  Z=14
=============================================================================
"""

import torch
import torch.nn as nn


class ConvBiLSTM(nn.Module):
    """
    ConvBiLSTM for 7-day fire spread prediction.

    Parameters
    ----------
    n_zones      : number of zones in the district
    n_features   : number of input features (65)
    n_timesteps  : lookback window (14)
    n_horizon    : forecast horizon (7)
    conv_filters : Conv1D output channels
    lstm_units   : hidden units per BiLSTM direction
    dropout      : dropout probability
    """

    def __init__(self,
                 n_zones: int,
                 n_features: int = 65,
                 n_timesteps: int = 14,
                 n_horizon: int = 7,
                 conv_filters: int = 32,
                 lstm_units: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        self.n_zones     = n_zones
        self.n_timesteps = n_timesteps
        self.n_horizon   = n_horizon

        # ── Block 1: Spatial Conv (applied at each timestep) ──────────────
        # Input per timestep: (batch, Z, F)
        # Conv1D slides over zone axis
        kernel = min(3, n_zones)
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(n_features, conv_filters, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_filters),
            nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_filters * 2),
            nn.Dropout(dropout),
        )
        conv_out_dim = conv_filters * 2 * n_zones   # flattened spatial output

        # ── Block 2: Bidirectional LSTM ────────────────────────────────────
        self.bilstm1 = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.bn_lstm1 = nn.BatchNorm1d(n_timesteps)

        self.bilstm2 = nn.LSTM(
            input_size=lstm_units * 2,
            hidden_size=lstm_units // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.dropout2 = nn.Dropout(dropout)

        lstm_out_dim = (lstm_units // 2) * 2   # bidirectional doubles hidden size

        # ── Block 3: Multi-step decoder ────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_units),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(lstm_units, n_horizon * n_zones),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, T, Z, F)
        returns : (batch, HORIZON, Z)  — raw logits (apply sigmoid for prob)
        """
        batch, T, Z, F = x.shape

        # Apply spatial conv at each timestep
        # Reshape: (batch*T, F, Z)  → Conv1D expects (N, C_in, L)
        x_conv = x.view(batch * T, Z, F).permute(0, 2, 1)   # (B*T, F, Z)
        x_conv = self.spatial_conv(x_conv)                   # (B*T, C_out, Z)
        x_conv = x_conv.flatten(1)                           # (B*T, C_out*Z)
        x_conv = x_conv.view(batch, T, -1)                   # (B, T, C_out*Z)

        # BiLSTM 1
        x_lstm, _ = self.bilstm1(x_conv)                    # (B, T, lstm*2)
        x_lstm = self.bn_lstm1(x_lstm)
        x_lstm = self.dropout1(x_lstm)

        # BiLSTM 2  — take last timestep output
        x_lstm, _ = self.bilstm2(x_lstm)                    # (B, T, lstm)
        x_last = x_lstm[:, -1, :]                           # (B, lstm)
        x_last = self.dropout2(x_last)

        # Decode → (B, HORIZON * Z)
        out = self.decoder(x_last)
        out = out.view(batch, self.n_horizon, self.n_zones)  # (B, H, Z)
        return out


def build_model(n_zones: int, device: torch.device, **kwargs) -> ConvBiLSTM:
    model = ConvBiLSTM(n_zones=n_zones, **kwargs)
    model.to(device)
    return model


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cpu")
    for name, n_zones in [("Banke", 14), ("Surkhet", 19)]:
        m = build_model(n_zones, device)
        x = torch.randn(8, 14, n_zones, 65)
        y = m(x)
        print(f"{name}: input {tuple(x.shape)} → output {tuple(y.shape)}  ✓")
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  Parameters: {total:,}")