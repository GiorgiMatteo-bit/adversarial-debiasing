"""
Gradient Reversal Layer for adversarial training.
Implements the mathematical operation:
- Forward: GRL(h) = h (identity)
- Backward: ∂L/∂h = -λ × ∂L_gender/∂GRL(h)
"""

import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from Ganin & Lempitsky (2015).
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass: identity operation.
        
        Args:
            x: input tensor [batch_size, feature_dim]
            lambda_: gradient reversal strength
        Returns:
            x: unchanged input
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse and scale gradients.
        
        Args:
            grad_output: gradients from subsequent layers
        Returns:
            reversed gradients, None (for lambda_)
        """
        lambda_ = ctx.lambda_
        return grad_output.neg() * lambda_, None


class GradientReversalLayer(torch.nn.Module):
    """Gradient Reversal Layer module."""
    
    def __init__(self, lambda_=1.0):
        """
        Args:
            lambda_: gradient reversal strength (default: 1.0)
        """
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, feature_dim]
        Returns:
            x: unchanged in forward, gradients reversed in backward
        """
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        """Update lambda parameter (used for progressive scheduling)."""
        self.lambda_ = lambda_
