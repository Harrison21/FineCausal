import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os


class FeatureGATCausal(nn.Module):
    """
    GAT-based causal intervention module for feature refinement
    Uses torch_geometric for efficient graph attention mechanisms
    """

    def __init__(self, feature_dim=1024, hidden_dim=512, num_heads=4, dropout=0.1):
        super(FeatureGATCausal, self).__init__()

        # GAT layers - remove the return_attention_weights parameter
        self.gat1 = GATv2Conv(
            in_channels=feature_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=feature_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim), nn.ReLU()
        )

        self.residual_weight = nn.Parameter(torch.FloatTensor([0.1]))
        self.feature_dim = feature_dim
        self.num_heads = num_heads

    def forward(
        self,
        video_1_fea,
        video_2_fea,
        video_1_fused,
        video_2_fused,
        mask_feature=None,
        return_attention=False,
    ):
        """
        Forward pass building a causal graph among features:
        - Original video features
        - Fused video features
        - Mask features (optional)

        Args:
            video_1_fea: Original video 1 features [B1, T, C]
            video_2_fea: Original video 2 features [B2, T, C]
            video_1_fused: Fused video 1 features [B1, T, C]
            video_2_fused: Fused video 2 features [B2, T, C]
            mask_feature: Mask features [B1+B2, T, C]
            return_attention: Whether to return attention weights

        Returns:
            Causally refined video features [B1+B2, T, C]
            and optionally attention weights
        """
        # Extract batch dimensions
        B1, T, C = video_1_fea.size()
        B2, _, _ = video_2_fea.size()

        # Create graph data structure
        all_features = []
        all_features.append(video_1_fea.reshape(B1 * T, C))  # Original video 1 features
        all_features.append(video_2_fea.reshape(B2 * T, C))  # Original video 2 features
        all_features.append(video_1_fused.reshape(B1 * T, C))  # Fused video 1 features
        all_features.append(video_2_fused.reshape(B2 * T, C))  # Fused video 2 features

        # Store node types for visualization
        node_types = (
            ["video_1"] * (B1 * T)
            + ["video_2"] * (B2 * T)
            + ["video_1_fused"] * (B1 * T)
            + ["video_2_fused"] * (B2 * T)
        )

        if mask_feature is not None:
            video_1_mask = mask_feature[:B1]
            video_2_mask = mask_feature[B1:]
            all_features.append(
                video_1_mask.reshape(B1 * T, C)
            )  # Mask features for video 1
            all_features.append(
                video_2_mask.reshape(B2 * T, C)
            )  # Mask features for video 2

            node_types += ["video_1_mask"] * (B1 * T) + ["video_2_mask"] * (B2 * T)

        # Concatenate all features to form graph nodes
        x = torch.cat(all_features, dim=0)  # [Total_Nodes, C]

        # Construct edge connections - fully connected graph
        num_nodes = x.size(0)
        edge_index = self._create_fully_connected_edges(num_nodes).to(x.device)

        # Apply GAT layers with return_attention_weights during forward call
        if return_attention:
            # For layer 1
            out1, attention1 = self.gat1(x, edge_index, return_attention_weights=True)
            out1 = F.dropout(out1, p=0.1, training=self.training)
            out1 = F.elu(out1)

            # For layer 2
            out2, attention2 = self.gat2(
                out1, edge_index, return_attention_weights=True
            )
            h = self.output(out2)

            # Store attention information
            edge_index_layer1, attn_weights_layer1 = attention1
            edge_index_layer2, attn_weights_layer2 = attention2
        else:
            # Standard forward pass without returning attention
            out1 = self.gat1(x, edge_index)
            out1 = F.dropout(out1, p=0.1, training=self.training)
            out1 = F.elu(out1)

            out2 = self.gat2(out1, edge_index)
            h = self.output(out2)

        # Apply residual connection with learnable weight
        h = h + self.residual_weight * x

        # Split back to get video 1 and video 2 features
        # Only take the first two parts which are the original features
        processed_video_1 = h[: B1 * T].view(B1, T, C)
        processed_video_2 = h[B1 * T : B1 * T + B2 * T].view(B2, T, C)

        # Concatenate processed features
        output = torch.cat([processed_video_1, processed_video_2], dim=0)

        if return_attention:
            # Process and return attention weights
            attention_data = {
                "node_types": node_types,
                "num_nodes": num_nodes,
                "layer1": {
                    "edge_index": edge_index_layer1,
                    "attn_weights": attn_weights_layer1,
                    "num_heads": self.num_heads,
                },
                "layer2": {
                    "edge_index": edge_index_layer2,
                    "attn_weights": attn_weights_layer2,
                    "num_heads": self.num_heads,
                },
                "B1": B1,
                "B2": B2,
                "T": T,
            }
            return output, attention_data

        return output

    def _create_fully_connected_edges(self, num_nodes):
        """Create edges for a fully-connected graph"""
        # Create all possible edges (except self-loops)
        source_nodes = torch.arange(num_nodes).repeat_interleave(num_nodes - 1)
        target_nodes = torch.cat(
            [
                torch.cat([torch.arange(i), torch.arange(i + 1, num_nodes)])
                for i in range(num_nodes)
            ]
        )
        edge_index = torch.stack([source_nodes, target_nodes], dim=0).long()
        return edge_index

    def visualize_attention(self, attention_data, save_path=None, head_idx=None):
        """
        Visualize the attention weights between different node types with extra large fonts

        Args:
            attention_data: Dictionary containing attention weights and metadata
            save_path: Path to save the visualization
            head_idx: Which attention head to visualize (for multi-head attention)
                    If None, will visualize average across all heads
        """
        # Extract data
        node_types = attention_data["node_types"]
        num_nodes = attention_data["num_nodes"]
        B1 = attention_data["B1"]
        B2 = attention_data["B2"]
        T = attention_data["T"]

        # We'll visualize the second GAT layer as it's the final attention mechanism
        edge_index = attention_data["layer2"]["edge_index"]
        attn_weights = attention_data["layer2"]["attn_weights"]
        num_heads = attention_data["layer2"]["num_heads"]

        # Group attention weights by node type
        node_type_groups = ["video_1", "video_2", "video_1_fused", "video_2_fused"]
        if len(node_types) > 4 * (B1 + B2) * T:
            node_type_groups += ["video_1_mask", "video_2_mask"]

        # Create the base directory if it doesn't exist
        save_dir = os.path.dirname(save_path) if save_path else "attention_maps"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set MUCH larger font sizes for better readability
        plt.rcParams.update(
            {
                "font.size": 20,  # Increased from 14
                "axes.titlesize": 24,  # Increased from 18
                "axes.labelsize": 22,  # Increased from 16
                "xtick.labelsize": 20,  # Increased from 14
                "ytick.labelsize": 20,  # Increased from 14
                "legend.fontsize": 20,  # Increased from 14
                "figure.titlesize": 26,  # Increased from 20
            }
        )

        # If head_idx is None, visualize the average across all heads
        if head_idx is None:
            # Average the attention weights across all heads
            if attn_weights.dim() > 1:
                avg_attn_weights = attn_weights.mean(dim=1)
            else:
                avg_attn_weights = attn_weights

            # Create a matrix to aggregate attention weights between node types
            node_type_matrix = np.zeros((len(node_type_groups), len(node_type_groups)))
            count_matrix = np.zeros_like(node_type_matrix)

            # Fill the matrix
            for i, (src, dst) in enumerate(edge_index.t().cpu().numpy()):
                src_type_idx = node_type_groups.index(node_types[src])
                dst_type_idx = node_type_groups.index(node_types[dst])
                node_type_matrix[src_type_idx, dst_type_idx] += avg_attn_weights[
                    i
                ].item()
                count_matrix[src_type_idx, dst_type_idx] += 1

            # Average the attention weights
            node_type_matrix = np.divide(
                node_type_matrix,
                count_matrix,
                out=np.zeros_like(node_type_matrix),
                where=count_matrix != 0,
            )

            # Plot the average heatmap with even larger figure size
            plt.figure(figsize=(16, 14))  # Increased from (14, 12)
            ax = sns.heatmap(
                node_type_matrix,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                xticklabels=node_type_groups,
                yticklabels=node_type_groups,
                annot_kws={"size": 20},  # Increased from 14
                cbar_kws={"label": "Attention Weight", "shrink": 0.8},
            )
            plt.title(
                f"Average Attention Weights Between Node Types\n(Averaged Across All Heads)"
            )
            plt.ylabel("Source Node Type", fontweight="bold")
            plt.xlabel("Target Node Type", fontweight="bold")

            # Improve tick label readability
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(pad=2.0)  # Add more padding

            # Save the averaged attention map
            averaged_save_path = os.path.join(
                save_dir, f"GAT_attention_average_all_heads_{timestamp}.png"
            )
            plt.savefig(averaged_save_path, bbox_inches="tight", dpi=300)
            plt.close()

            return node_type_matrix

        # If head_idx is provided, visualize that specific head
        else:
            # Select the specified attention head
            if attn_weights.dim() > 1:
                head_attn_weights = attn_weights[:, head_idx]
            else:
                head_attn_weights = attn_weights

            # Create a matrix to aggregate attention weights between node types
            node_type_matrix = np.zeros((len(node_type_groups), len(node_type_groups)))
            count_matrix = np.zeros_like(node_type_matrix)

            # Fill the matrix
            for i, (src, dst) in enumerate(edge_index.t().cpu().numpy()):
                src_type_idx = node_type_groups.index(node_types[src])
                dst_type_idx = node_type_groups.index(node_types[dst])
                node_type_matrix[src_type_idx, dst_type_idx] += head_attn_weights[
                    i
                ].item()
                count_matrix[src_type_idx, dst_type_idx] += 1

            # Average the attention weights
            node_type_matrix = np.divide(
                node_type_matrix,
                count_matrix,
                out=np.zeros_like(node_type_matrix),
                where=count_matrix != 0,
            )

            # Plot the heatmap with larger figure size
            plt.figure(figsize=(16, 14))  # Increased from (14, 12)
            ax = sns.heatmap(
                node_type_matrix,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                xticklabels=node_type_groups,
                yticklabels=node_type_groups,
                annot_kws={"size": 20},  # Increased from 14
                cbar_kws={"label": "Attention Weight", "shrink": 0.8},
            )
            plt.title(
                f"Average Attention Weights Between Node Types\n(Head {head_idx})"
            )
            plt.ylabel("Source Node Type", fontweight="bold")
            plt.xlabel("Target Node Type", fontweight="bold")

            # Improve tick label readability
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(pad=2.0)  # Add more padding

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()

            return node_type_matrix
