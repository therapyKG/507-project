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
import utils

# ============================================================================
#                      CONFIGURATION
# ============================================================================

# --- Model Config ---
MODEL_NAME = 'gpt2-medium'       # Base model to extract weights from
TILE_SIZE = 256           # Tile H/W
PATCH_SIZE = 16           # Patch size for ViT
COMPRESSED_DIM = 64       # 'k' dimension (latent dim for each patch)

# --- ViT Encoder Config ---
VIT_DEPTH = 24             # Number of Transformer blocks in ViT
VIT_HEADS = 16             # Number of heads in ViT
VIT_MLP_RATIO = 16.0       # MLP ratio inside ViT blocks

# --- New Decoder Config ---
# We use a deep MLP to give it combinatorial capacity to memorize
DECODER_HIDDEN_DIM = 2048 # Hidden dim for the deep MLP decoder

# --- Training Config ---
AE_EPOCHS = 1000         
AE_BATCH_SIZE = 64        # Autoencoder batch size
AE_LEARNING_RATE = 1e-4   # Learning rate for autoencoder
GRAD_CLIP_VALUE = 1.0     # Max norm for gradient clipping

# --- Perplexity Config ---
PPL_STRIDE = 512          # Stride for perplexity calculation
PPL_MAX_LENGTH = 1024     # Max sequence length for perplexity

# ============================================================================
#               MODEL RECONSTRUCTION & PERPLEXITY
# ============================================================================

@torch.no_grad()
def reconstruct_gpt2(autoencoder, model_name, tile_metadata, device):
    """
    Creates a new GPT-2 model with MLP weights reconstructed by the AE.
    """
    print(f"\nLoading fresh '{model_name}' model for reconstruction...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

    autoencoder.eval()
    autoencoder.to(device)
    
    print("Reconstructing and replacing MLP layers...")
    
    replaced_layers = 0
    for name, module in model.named_modules():
        if name in tile_metadata:
            metadata = tile_metadata[name]
            original_shape = metadata['original_shape']
            
            # Get original weight, transpose, and tile it
            weight_t = module.weight.data.T.clone().contiguous()
            tiles, _ = utils.tile_matrix(weight_t, autoencoder.tile_size)
            
            if not tiles:
                print(f"  - WARNING: No tiles generated for '{name}'. Skipping.")
                continue

            tiles_tensor = torch.stack(tiles).to(device)
            
            # Reconstruct tiles in batches to save VRAM
            recon_tiles = []
            for batch in tiles_tensor.split(AE_BATCH_SIZE):
                recon_batch = autoencoder(batch)
                recon_tiles.append(recon_batch.cpu())
            
            recon_tiles_tensor = torch.cat(recon_tiles, dim=0)

            # Reconstruct the full weight matrix
            recon_weight_t = utils.reconstruct_from_tiles(
                recon_tiles_tensor, 
                metadata, 
                autoencoder.tile_size,
                device='cpu'
            )
            
            # Transpose back to (out_features, in_features)
            recon_weight = recon_weight_t.T.contiguous()
            
            # Check shape match before assignment
            if recon_weight.shape == module.weight.data.shape:
                module.weight.data = recon_weight
                replaced_layers += 1
            else:
                print(f"  - ERROR: Shape mismatch for '{name}'.")
                print(f"    Expected: {module.weight.data.shape}")
                print(f"    Got:      {recon_weight.shape}")
            
            del weight_t, tiles, tiles_tensor, recon_tiles, recon_tiles_tensor, recon_weight
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Replaced {replaced_layers} MLP layers.")
    return model

@torch.no_grad()
def calculate_perplexity(model, tokenizer, device):
    """
    Calculates perplexity on the wikitext-2 test set.
    """
    print("Loading test dataset 'wikitext-2-raw-v1'...")
    try:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure you have an internet connection.")
        return -1.0
        
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    model.eval()
    model.to(device)
    
    nlls = []
    seq_len = encodings.input_ids.size(1)
    
    ppl_pbar = tqdm(range(0, seq_len, PPL_STRIDE), desc="Calculating PPL", unit="stride")
    for begin_loc in ppl_pbar:
        end_loc = min(begin_loc + PPL_MAX_LENGTH, seq_len)
        trg_len = end_loc - begin_loc
        
        # Stop if sequence is too short
        if trg_len <= 1:
            break
            
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        
        # Target labels are the same as input_ids
        target_ids = input_ids.clone()
        
        # We don't want to calculate loss on padding
        # We only care about the loss of the new tokens
        if end_loc < seq_len:
             # Set tokens from the overlapping region to -100
             target_ids[:, :-PPL_STRIDE] = -100
        
        try:
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
            if not math.isnan(neg_log_likelihood):
                nlls.append(neg_log_likelihood)
                ppl_pbar.set_postfix(nll=f"{neg_log_likelihood.item():.4f}")
            else:
                print("Warning: NaN loss detected.")

        except Exception as e:
            print(f"Error during perplexity calculation: {e}")
            break # Stop PPL calculation
            
        del input_ids, target_ids, outputs
        torch.cuda.empty_cache()

    if not nlls:
        print("Error: No valid NLL values recorded. Cannot compute perplexity.")
        return -1.0

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()