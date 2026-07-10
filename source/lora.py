import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA_Config:
    """
    Configuration class for Low-Rank Adaptation (LoRA) parameters.
    
    Args:
        r: Rank of the low-rank decomposition
        lora_alpha: Scaling factor for LoRA updates
        lora_dropout: Dropout probability for LoRA layers
        merge_weights: Whether to merge LoRA weights with original weights
        target_modules: List of module names to apply LoRA to
    """
    def __init__(self, r, lora_alpha, lora_dropout, merge_weights, target_modules):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        self.target_modules = target_modules

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that wraps an original linear layer.
    
    Implements the LoRA method by adding low-rank matrices A and B such that
    the effective weight becomes: W' = W + scaling * A @ B
    
    Args:
        original_layer: The original nn.Linear layer to adapt
        config: LoRA_Config object containing hyperparameters
    """
    def __init__(self, original_layer, config: LoRA_Config):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        input_dim = original_layer.weight.size(1)
        output_dim = original_layer.weight.size(0)

        lora_A_tensor = torch.empty(input_dim, config.r)
        torch.nn.init.kaiming_uniform_(lora_A_tensor)
        self.lora_A = nn.Parameter(lora_A_tensor)

        self.lora_B = nn.Parameter(torch.zeros(config.r, output_dim))
        self.scaling = config.lora_alpha/config.r
        self.bias = original_layer.bias

        if config.lora_dropout > 0:
            self.dropout = nn.Dropout(p=config.lora_dropout)
        else:
            self.dropout = lambda x: x  # No-op
    @property
    def weight(self):
        """
        Property that redirects weight access to the original layer's weight.
        
        Returns:
            The original layer's weight tensor
        """
        return self.original_layer.weight

    def forward(self, x):
        """
        Forward pass applying LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying W' = W + scaling * dropout(A) @ dropout(B)
        """
        A_dropout = self.dropout(self.lora_A)
        B_dropout = self.dropout(self.lora_B)
        W_prime = self.original_layer.weight + self.scaling*A_dropout @ B_dropout
        return F.linear(x, W_prime, self.original_layer.bias)
    def __repr__(self):
        return f'{self.__class__.__name__}(\n  (original_layer): {self.original_layer},\n  (lora_A): Parameter of size {self.lora_A.size()},\n  (lora_B): Parameter of size {self.lora_B.size()}\n)'

def print_trainable_parameters(model):
    """
    Print the number and percentage of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, all_params)
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:  # count only learnable parameters
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param:.2f} %"
    )
    return trainable_params, all_param

def apply_lora_to_model(model, config):
    """
    Apply LoRA adaptation to specified modules in the model.
    
    Args:
        model: PyTorch model to apply LoRA to
        config: LoRA_Config object specifying which modules to adapt
        
    Returns:
        Modified model with LoRA layers
    """
    for name, module in model.named_modules():
        hierarchy = name.split('.')
        if len(hierarchy) > 1:  # Ensure the module is not the top-level module
            parent_module = model
            for submodule_name in hierarchy[:-1]:  # Navigate to the parent module
                parent_module = getattr(parent_module, submodule_name)
            
            layer_name = hierarchy[-1]
            for target_module in config.target_modules:
                if target_module in layer_name:
                    original_layer = getattr(parent_module, layer_name)
                    if isinstance(original_layer, nn.Linear):
                        setattr(parent_module, layer_name, LoRALayer(original_layer, config))
    return model
    
# Freeze everything except the LoRA layers (and, optionally, biases).
def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    Mark only LoRA parameters as trainable, freezing all other parameters.
    
    Args:
        model: PyTorch model with LoRA layers
        bias: Bias handling strategy - 'none', 'all', or 'lora_only'
              - 'none': No bias parameters are trainable
              - 'all': All bias parameters are trainable
              - 'lora_only': Only LoRA layer biases are trainable
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError