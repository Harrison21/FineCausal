import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CausalModelConfig:
    def __init__(self, mask_feature_size):
        self.fusion_mode = "add"

        # Get dimensions from tensor sizes
        C = mask_feature_size

        self.mask_layers = type("", (), {})()
        self.mask_layers.d_in = C  # Dynamic input size
        self.mask_layers.d_hidden = 2 * C  # Hidden size based on factor
        self.mask_layers.d_out = C  # Keep channel dimension

        self.com_feature_layers = type("", (), {})()
        self.com_feature_layers.d_in = C  # Dynamic input size
        self.com_feature_layers.d_hidden = 2 * C  # Hidden size based on factor
        self.com_feature_layers.d_out = C  # Keep channel dimension


class CausalIntervention(nn.Module):
    def __init__(self, causal_model_config):
        super().__init__()
        self.fusion_mode = causal_model_config.fusion_mode

        # Mlp for mask_feature
        self.mask_layers = Mlp(
            in_features=causal_model_config.mask_layers.d_in,
            hidden_features=causal_model_config.mask_layers.d_hidden,
            out_features=causal_model_config.mask_layers.d_out,
            act_layer=nn.GELU,  # Default activation in Mlp
            drop=0.1,  # Adjust dropout if needed
        )

        # Mlp for com_feature (either com_feature_12_u or com_feamap_12_u)
        self.com_feature_layers = Mlp(
            in_features=causal_model_config.com_feature_layers.d_in,
            hidden_features=causal_model_config.com_feature_layers.d_hidden,
            out_features=causal_model_config.com_feature_layers.d_out,
            act_layer=nn.GELU,  # Default activation in Mlp
            drop=0.0,  # Adjust dropout if needed
        )

    def forward(self, mask_feature, com_feature):
        # Ensure same device
        device = mask_feature.device
        self.mask_layers = self.mask_layers.to(device)
        self.com_feature_layers = self.com_feature_layers.to(device)

        # Process inputs independently
        mask_out = self.mask_layers(mask_feature)
        com_feature_out = self.com_feature_layers(com_feature)

        # Fusion logic
        if self.fusion_mode == "add":
            fused_feature = 0.5 * mask_out + 0.5 * com_feature_out
        elif self.fusion_mode == "concat":
            fused_feature = torch.cat((mask_out, com_feature_out), dim=1)
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        return fused_feature
