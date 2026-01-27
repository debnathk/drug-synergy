"""
Description: Kolmogorov-Arnold Network (KAN) implementation using B-spline basis functions.
             KAN replaces traditional linear layers with learnable univariate spline functions.
Author: Kusal Debnath
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    KAN Linear layer using B-spline basis functions.
    
    Instead of y = Wx + b (MLP), KAN computes:
    y_j = sum_i phi_{i,j}(x_i)
    
    where phi_{i,j} are learnable univariate functions represented as B-splines.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1, 1),
    ):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            grid_size: Number of grid intervals for B-splines
            spline_order: Order of B-spline (3 = cubic)
            scale_noise: Noise scale for weight initialization
            scale_base: Scale for base weight initialization
            scale_spline: Scale for spline weight initialization
            enable_standalone_scale_spline: Whether to use separate spline scale parameter
            base_activation: Activation function for base path
            grid_eps: Small epsilon for grid extension
            grid_range: Range for grid points
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid for B-spline basis (extended for boundary conditions)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        
        # Base weight (linear transformation with activation) - provides stability
        self.base_weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        
        # Spline weights - learnable coefficients for B-spline basis
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # Optional separate scale for spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights."""
        # Base weight initialization (Kaiming)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        # Spline weight initialization with noise
        with torch.no_grad():
            noise = (
                (torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                self.scale_spline * self.curve2coeff(
                    torch.linspace(-1, 1, self.in_features).unsqueeze(0),
                    noise
                )
            )
            
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
    
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            B-spline basis of shape [batch_size, in_features, grid_size + spline_order]
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        grid = self.grid  # [in_features, grid_size + 2*spline_order + 1]
        x = x.unsqueeze(-1)  # [batch, in_features, 1]
        
        # B-spline recursion (Cox-de Boor algorithm)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute spline coefficients given sample points.
        
        Args:
            x: Input sample points [batch, in_features]
            y: Output values [out_features, in_features, grid_size + spline_order]
            
        Returns:
            Spline coefficients
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        # Compute basis at sample points
        A = self.b_splines(x).transpose(0, 1)  # [in_features, batch, grid_size + spline_order]
        
        # Solve least squares for coefficients
        B = y.transpose(0, 1)  # [in_features, out_features, grid_size + spline_order]
        
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.transpose(0, 1)  # [out_features, in_features, grid_size + spline_order]
        
        return result.contiguous()
    
    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """Get scaled spline weights."""
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        
        # Base path: linear transformation with activation
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Spline path: B-spline basis weighted by learned coefficients
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        output = base_output + spline_output
        
        # Restore original batch dimensions
        output = output.view(*original_shape[:-1], self.out_features)
        return output


class SynergyKAN(nn.Module):
    """
    KAN-based model for drug synergy prediction.
    Uses smaller layer sizes than MLP due to KAN's higher expressiveness.
    
    Architecture: 2304 -> 128 -> 32 -> 1
    """
    
    def __init__(self, input_dim: int = 2304, output_dim: int = 1, grid_size: int = 5):
        """
        Args:
            input_dim: Input dimension (default 2304 = 3 * 768 for drug1 + drug2 + omics)
            output_dim: Output dimension (default 1 for synergy score)
            grid_size: Grid size for B-spline basis
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            KANLinear(input_dim, 128, grid_size=grid_size),
            KANLinear(128, 32, grid_size=grid_size),
            KANLinear(32, output_dim, grid_size=grid_size),
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layers.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x
