from ConvLSTM import ConvLSTM
import torch
import torch.nn as nn
from collections import defaultdict

#MLP definition
class MLP_5D(nn.Module):
    def __init__(self, height, width):
        super(MLP_5D, self).__init__()
        # Define the fully connected layers
        self.fc1 = nn.Linear(64, 128)  # Input channels = 41, output features = 128
        self.dropout1 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(128, 64)  # Output features = 64
        self.dropout2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(64, 1)    # Final output, reducing to 1 channel

        self.height = height
        self.width = width

    def forward(self, x):
        batch_size, timesteps, channels, height, width = x.shape
        
        # Ensure the input spatial dimensions match the expected height and width
        assert height == self.height and width == self.width, "Height and width mismatch"
        
        # Reshape to (batch * timesteps * height * width, channels)
        x = x.permute(0, 1, 3, 4, 2).reshape(-1, channels)
        # print(x.shape)
        
        # Apply MLP (Fully connected layers)
        x = self.fc1(x)
        x = torch.nn.functional.softplus(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.nn.functional.softplus(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.nn.functional.softplus(x)
        
        # Reshape back to (batch, timesteps, 1, height, width)
        x = x.view(batch_size, timesteps, self.height, self.width, 1).permute(0, 1, 4, 2, 3)

        return x
    
  
# MultiTask ConvLSTM definition

class ConvLSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, output_channels, batch_first=True, pool_size=(2,2)):
        super(ConvLSTMNetwork, self).__init__()
        
        # ConvLSTM module
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dim=hidden_dims,
                                 kernel_size=kernel_size,
                                 num_layers=num_layers,
                                 batch_first=batch_first,
                                 bias=True,
                                 return_all_layers=True)
        
        # Batch Normalization for each ConvLSTM layer's output
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm3d(hidden_dim) for hidden_dim in hidden_dims
        ])

        # Final Conv3D layer for regression pathway
        self.conv3d = nn.Conv3d(in_channels=hidden_dims[-1],
                                out_channels=output_channels,
                                kernel_size=(1, 3, 3),
                                padding=(0, 1, 1))

        # MLP for regression output: (B,T,C,H,W) -> (B,T,1,H,W)
        self.mlp = MLP_5D(height=81, width=97)

        # Classification head for pixel-level zero precipitation probability
        # We'll produce (B,T,1,H,W) as well:
        # The classification head takes (B,C,T,H,W) input. We'll reorder dimensions before applying it.
        # Then apply Sigmoid to get probabilities between 0 and 1.
        self.classification_head = nn.Sequential(
            nn.Conv3d(output_channels, 1, kernel_size=(1,1,1)),  # from C to 1 channel
            nn.Sigmoid()
        )

        self.activation_variance = defaultdict(list)

    def forward(self, x):
        """
        x: (B, T, input_dim, H, W)
        """
        # Forward through ConvLSTM
        layer_output_list, last_state_list = self.convlstm(x)
        
        # Apply batch norms
        for i, output in enumerate(layer_output_list):
            # output: (B, T, C, H, W)
            output = output.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) for BatchNorm3d
            output = self.batch_norms[i](output)
            output = output.permute(0, 2, 1, 3, 4)  # back to (B, T, C, H, W)

            #Track variance across spatial dimensions for hooks with activation tracking 
            activation_variance = output.var(dim=(3, 4)).mean().item()
            self.activation_variance[f"ConvLSTM_layer_{i}"].append(activation_variance)

            layer_output_list[i] = output
        
        # Take output from the last ConvLSTM layer
        final_output = layer_output_list[-1]  # (B, T, C, H, W)

        # Pass through Conv3D: needs (B,C,T,H,W)
        final_output = final_output.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W)
        final_output = self.conv3d(final_output)
        # Now final_output: (B, output_channels, T, H, W)

        # Return to (B,T,C,H,W) for MLP (regression)
        final_output_t = final_output.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W)

        # Regression output
        regression_output = self.mlp(final_output_t)  # (B,T,1,H,W)

        # Classification output:
        # The classification head is defined for (B,C,T,H,W), so reorder again
        final_output_c = final_output  # still (B,output_channels,T,H,W)
        classification_output = self.classification_head(final_output_c)
        # classification_output: (B,1,T,H,W)

        # Permute classification output to match (B,T,1,H,W) format
        classification_output = classification_output.permute(0, 2, 1, 3, 4)  # (B,T,1,H,W)

        return regression_output, classification_output