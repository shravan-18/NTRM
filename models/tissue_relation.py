"""
Neural Tissue Relation Modeling (NTRM) module.

This module implements a novel approach to histopathology segmentation
by explicitly modeling relationships between different tissue types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionProposalNetwork(nn.Module):
    """
    Extracts region proposals from segmentation
    """
    def __init__(self, in_channels, out_channels=128):
        super(RegionProposalNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, features, segmentation):
        """
        Args:
            features: CNN features [B, C, H, W]
            segmentation: Initial segmentation [B, num_classes, H, W]
            
        Returns:
            regions: Dictionary of region proposals per class
        """
        batch_size, num_classes = segmentation.shape[0], segmentation.shape[1]
        regions = {}
        
        # Process features
        refined_features = self.relu(self.bn(self.conv(features)))
        
        # For each class, create region proposals
        for c in range(num_classes):
            # Get binary mask for this class
            class_mask = segmentation[:, c:c+1]  # [B, 1, H, W]
            
            # Apply morphological operations (approximated with max pooling followed by thresholding)
            pooled_mask = F.max_pool2d(class_mask, kernel_size=3, stride=1, padding=1)
            refined_mask = (pooled_mask > 0.5).float()
            
            # Apply features to mask
            masked_features = refined_features * refined_mask
            
            # Store in regions dictionary
            regions[c] = {
                'mask': refined_mask,
                'features': masked_features
            }
            
        return regions


class NodeFeatureExtractor(nn.Module):
    """
    Extracts node features for tissue regions
    """
    def __init__(self, in_channels, hidden_dim=64):
        super(NodeFeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        
    def forward(self, regions):
        """
        Args:
            regions: Dictionary of region proposals
            
        Returns:
            node_features: Features for each tissue type [B, num_classes, hidden_dim]
            node_masks: Binary masks for each node [B, num_classes, H, W]
        """
        batch_size = next(iter(regions.values()))['mask'].shape[0]
        num_classes = len(regions)
        hidden_dim = self.conv.out_channels
        h, w = next(iter(regions.values()))['mask'].shape[2], next(iter(regions.values()))['mask'].shape[3]
        
        # Initialize output tensors
        node_features = torch.zeros(batch_size, num_classes, hidden_dim, device=next(iter(regions.values()))['mask'].device)
        node_masks = torch.zeros(batch_size, num_classes, h, w, device=next(iter(regions.values()))['mask'].device)
        
        # Process each region
        for c in range(num_classes):
            region = regions[c]
            mask = region['mask']  # Shape: [B, 1, H, W]
            features = region['features']
            
            # Transform features
            transformed = self.relu(self.bn(self.conv(features)))
            
            # Global average pooling with mask
            masked_sum = torch.sum(transformed * mask, dim=(2, 3))
            mask_sum = torch.sum(mask, dim=(2, 3)) + 1e-6  # Avoid division by zero
            pooled = masked_sum / mask_sum
            
            # Store results
            node_features[:, c] = pooled
            node_masks[:, c] = mask.reshape(batch_size, h, w)  # Reshape instead of squeeze
            
        return node_features, node_masks


class EdgeFeatureExtractor(nn.Module):
    """
    Extracts edge features between tissue regions
    """
    def __init__(self, hidden_dim=64):
        super(EdgeFeatureExtractor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_features, node_masks):
        """
        Args:
            node_features: Features for each tissue type [B, num_classes, hidden_dim]
            node_masks: Binary masks for each node [B, num_classes, H, W]
            
        Returns:
            edge_index: List of edge indices for each batch [B, 2, E]
            edge_attr: Edge features for each batch [B, E, hidden_dim]
        """
        batch_size, num_classes = node_features.shape[0], node_features.shape[1]
        hidden_dim = node_features.shape[2]
        
        all_edge_indices = []
        all_edge_attrs = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Initialize edge storage
            edge_indices = []
            edge_attrs = []
            
            # Calculate adjacency from masks
            masks_b = node_masks[b]  # [num_classes, H, W]
            
            # Create edges between all classes
            for i in range(num_classes):
                for j in range(num_classes):
                    if i != j:  # No self-loops
                        # Check if regions are adjacent (can be done better with proper morphological operations)
                        mask_i = masks_b[i]
                        mask_j = masks_b[j]
                        
                        # Only add edge if both masks have some content
                        if torch.sum(mask_i) > 0 and torch.sum(mask_j) > 0:
                            edge_indices.append([i, j])
                            
                            # Create edge features by concatenating node features
                            node_i_feat = node_features[b, i]
                            node_j_feat = node_features[b, j]
                            edge_feat = torch.cat([node_i_feat, node_j_feat])
                            
                            # Process edge features through MLP
                            edge_attr = self.mlp(edge_feat)
                            edge_attrs.append(edge_attr)
            
            # Convert to tensors
            if edge_indices:
                edge_index = torch.tensor(edge_indices, device=node_features.device).t()  # [2, E]
                edge_attr = torch.stack(edge_attrs)  # [E, hidden_dim]
            else:
                # Create dummy edge if no edges found
                edge_index = torch.zeros((2, 1), device=node_features.device, dtype=torch.long)
                edge_attr = torch.zeros((1, hidden_dim), device=node_features.device)
                
            all_edge_indices.append(edge_index)
            all_edge_attrs.append(edge_attr)
            
        return all_edge_indices, all_edge_attrs


class GraphConvolutionLayer(nn.Module):
    """
    Simple graph convolution layer
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, in_features]
            
        Returns:
            output: Updated node features [num_nodes, out_features]
        """
        src, dst = edge_index
        
        # Transform node features
        x = torch.matmul(x, self.weight)
        
        # Prepare output
        out = torch.zeros_like(x)
        
        # Aggregate messages from neighbors
        for i in range(edge_index.size(1)):
            s, d = src[i], dst[i]
            if edge_attr is not None:
                # Use edge features to weight message
                message = x[s] * edge_attr[i]
            else:
                message = x[s]
                
            # Add message to destination node
            out[d] += message
            
        # Add bias
        out += self.bias
        
        return out


class TissueGraphNetwork(nn.Module):
    """
    Graph neural network for modeling tissue relationships
    """
    def __init__(self, hidden_dim=64, num_classes=12, num_layers=3, enable_global_embeddings=True):
        super(TissueGraphNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.enable_global_embeddings = enable_global_embeddings
        
        # Graph convolution layers
        self.gconvs = nn.ModuleList([
            GraphConvolutionLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Layer normalization after each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Global tissue knowledge embeddings
        if enable_global_embeddings:
            self.global_tissue_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
            nn.init.normal_(self.global_tissue_embeddings, mean=0.0, std=0.02)
        
    def forward(self, node_features, edge_indices, edge_attrs):
        """
        Args:
            node_features: Features for each tissue type [B, num_classes, hidden_dim]
            edge_indices: List of edge indices for each batch [B, 2, E]
            edge_attrs: Edge features for each batch [B, E, hidden_dim]
            
        Returns:
            updated_node_features: Updated node features [B, num_classes, hidden_dim]
        """
        batch_size = node_features.shape[0]
        updated_features = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Get features for this batch
            x = node_features[b]  # [num_classes, hidden_dim]
            edge_index = edge_indices[b]  # [2, E]
            edge_attr = edge_attrs[b]  # [E, hidden_dim]
            
            # Create presence mask
            presence_mask = (torch.sum(x, dim=1) != 0).float().unsqueeze(1)  # [num_classes, 1]
            
            # Apply graph convolutions with residual connections
            for i, (gconv, norm) in enumerate(zip(self.gconvs, self.layer_norms)):
                # Apply graph convolution
                residual = x
                x = gconv(x, edge_index, edge_attr)
                x = F.relu(x)
                x = norm(x)
                
                # Add residual connection after first layer
                if i > 0:
                    x = x + residual
            
            # For missing tissues, use global embeddings if enabled
            if self.enable_global_embeddings:
                x = x * presence_mask + self.global_tissue_embeddings * (1 - presence_mask)
                
            updated_features.append(x)
            
        # Stack batch dimension back
        updated_features = torch.stack(updated_features)  # [B, num_classes, hidden_dim]
        
        return updated_features


class SpatialProjector(nn.Module):
    """
    Projects node features back to spatial domain
    """
    def __init__(self, hidden_dim=64, out_channels=128):
        super(SpatialProjector, self).__init__()
        self.conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, node_features, node_masks):
        """
        Args:
            node_features: Node features [B, num_classes, hidden_dim]
            node_masks: Binary masks for each node [B, num_classes, H, W]
            
        Returns:
            spatial_features: Features in spatial domain [B, out_channels, H, W]
        """
        batch_size, num_classes, hidden_dim = node_features.shape
        h, w = node_masks.shape[2], node_masks.shape[3]
        
        # Initialize output tensor
        spatial_features = torch.zeros(batch_size, hidden_dim, h, w, device=node_features.device)
        
        # Project each node's features to its spatial region
        for b in range(batch_size):
            for c in range(num_classes):
                # Get node features and mask
                feat = node_features[b, c]  # [hidden_dim]
                mask = node_masks[b, c]  # [H, W]
                
                # Broadcast features to mask region
                feat_expanded = feat.view(-1, 1, 1).expand(-1, h, w)  # [hidden_dim, H, W]
                spatial_features[b] += feat_expanded * mask
        
        # Apply final convolution
        output = self.relu(self.bn(self.conv(spatial_features)))
        
        return output


class TissueRelationModule(nn.Module):
    """
    Complete tissue relation modeling module
    """
    def __init__(self, in_channels, out_channels, num_classes=12, hidden_dim=64, 
                 gnn_layers=3, enable_global_embeddings=True):
        super(TissueRelationModule, self).__init__()
        
        # Components
        self.rpn = RegionProposalNetwork(in_channels, hidden_dim)
        self.node_extractor = NodeFeatureExtractor(hidden_dim, hidden_dim)
        self.edge_extractor = EdgeFeatureExtractor(hidden_dim)
        self.gnn = TissueGraphNetwork(hidden_dim, num_classes, gnn_layers, enable_global_embeddings)
        self.projector = SpatialProjector(hidden_dim, out_channels)
        
    def forward(self, features, initial_seg):
        """
        Args:
            features: CNN features [B, C, H, W]
            initial_seg: Initial segmentation [B, num_classes, H, W]
            
        Returns:
            enhanced_features: Enhanced features [B, out_channels, H, W]
        """
        # Generate region proposals
        regions = self.rpn(features, initial_seg)
        
        # Extract node features
        node_features, node_masks = self.node_extractor(regions)
        
        # Extract edge features
        edge_indices, edge_attrs = self.edge_extractor(node_features, node_masks)
        
        # Apply graph neural network
        updated_node_features = self.gnn(node_features, edge_indices, edge_attrs)
        
        # Project back to spatial domain
        enhanced_features = self.projector(updated_node_features, node_masks)
        
        return enhanced_features
