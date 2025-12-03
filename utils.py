import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Import Conv1D for correctly identifying MLP layers
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from tqdm import tqdm

import math
import gc
import copy
import time

# ============================================================================
#                      TILING UTILITIES (from attn.py)
# ============================================================================

def tile_matrix(matrix, tile_size=256):
    """
    Break a matrix into tiles with padding if necessary.
    Returns: list of tiles, original shape, padding info
    """
    h, w = matrix.shape
    
    # Calculate padding needed
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    # Pad matrix if needed
    if pad_h > 0 or pad_w > 0:
        # Pad with zeros
        matrix_padded = F.pad(matrix, (0, pad_w, 0, pad_h), 'constant', 0)
    else:
        matrix_padded = matrix
    
    padded_h, padded_w = matrix_padded.shape
    
    # Split into tiles
    tiles = []
    tile_positions = []
    
    for i in range(0, padded_h, tile_size):
        for j in range(0, padded_w, tile_size):
            tile = matrix_padded[i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
            tile_positions.append((i, j))
    
    # Metadata for reconstruction
    metadata = {
        'original_shape': (h, w),
        'padded_shape': (padded_h, padded_w),
        'tile_positions': tile_positions,
        'padding': (pad_h, pad_w),
        'num_tiles': len(tiles)
    }
    
    return tiles, metadata

def reconstruct_from_tiles(tiles, metadata, tile_size=256, device='cpu'):
    """
    Reconstruct matrix from tiles using metadata.
    """
    padded_h, padded_w = metadata['padded_shape']
    h, w = metadata['original_shape']
    tile_positions = metadata['tile_positions']
    
    # Create empty padded matrix
    reconstructed = torch.zeros(padded_h, padded_w, device=device)
    
    # Place tiles
    for tile, (i, j) in zip(tiles, tile_positions):
        reconstructed[i:i+tile_size, j:j+tile_size] = tile.squeeze()
    
    # Remove padding
    reconstructed = reconstructed[:h, :w]
    
    return reconstructed

# ============================================================================
#                      DATASET PREPARATION
# ============================================================================

def get_gpt2_mlp_tiles(model_name, tile_size):
    """
    Extracts all MLP layer weights from a GPT-2 model and tiles them.
    Uses the robust `tile_matrix` function.
    """
    print(f"Loading base model '{model_name}' to extract weight tiles...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None
        
    all_tiles = []
    all_metadata = {}
    total_tile_count = 0
    
    print("Extracting and tiling MLP layer weights...")
    for name, module in model.named_modules():
        # Correctly identify MLP layers (Conv1D)
        if isinstance(module, Conv1D) and ('mlp.c_fc' in name or 'mlp.c_proj' in name):
            
            # .weight is (out_features, in_features)
            # We transpose to (in_features, out_features)
            weight = module.weight.data.T.clone().contiguous()
            
            # Tile the weight matrix
            tiles, metadata = tile_matrix(weight, tile_size)
            
            all_tiles.extend(tiles)
            all_metadata[name] = metadata
            total_tile_count += len(tiles)
            
            print(f"  - Extracted '{name}': {tuple(weight.shape)} -> {len(tiles)} tiles")
            del weight, tiles
    
    del model
    gc.collect()

    if total_tile_count == 0:
        print("\nError: No MLP layers found. Check model name or layer names.")
        return None, None
        
    print(f"\nTotal tiles extracted: {total_tile_count}")
    
    try:
        # Stack all tiles into a single tensor
        stacked_tiles = torch.stack(all_tiles)
    except Exception as e:
        print("\nError: Could not stack tiles.")
        print(f"This can happen if tile_matrix produced non-uniform tiles.")
        print(f"Error: {e}")
        return None, None

    return stacked_tiles, all_metadata