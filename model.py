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
#                      ViT AUTOENCODER MODEL
# ============================================================================

class ViTAutoencoder(nn.Module):
    """
    Autoencoder using a ViT to encode and a deep MLP to decode.
    """
    def __init__(self, tile_size, patch_size, embed_dim, depth, num_heads, 
                 mlp_ratio, decoder_hidden_dim):
        super().__init__()
        
        """
        self.encoder = ViTEncoder(
            tile_size=tile_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        """

        self.encoder = nn.Sequential(
            nn.Linear(tile_size, decoder_hidden_dim),
            nn.ReLU(),
            #nn.LayerNorm(decoder_hidden_dim), # Add LayerNorm for stability
            
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            #nn.LayerNorm(decoder_hidden_dim),
            
            # Project back to one row of the tile
            nn.Linear(decoder_hidden_dim, embed_dim)
        )
        
        # --- NEW: Deep MLP Decoder ---
        # This gives the model combinatorial power to overfit/memorize
        self.decoder_mlp = nn.Sequential(
            nn.Linear(embed_dim, decoder_hidden_dim),
            nn.ReLU(),
            #nn.LayerNorm(decoder_hidden_dim), # Add LayerNorm for stability
            
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            #nn.LayerNorm(decoder_hidden_dim),
            
            # Project back to one row of the tile
            nn.Linear(decoder_hidden_dim, tile_size)
        )
        # --- END NEW ---

        self.tile_size = tile_size
        self.embed_dim = embed_dim

    def encode(self, x):
        # x shape: [B, 256, 256]
        # latent shape: [B, 256, 16]
        return self.encoder(x)

    def decode(self, latent):

        reconstructed = self.decoder_mlp(latent)
        
        return reconstructed

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

# ============================================================================
#                      AUTOENCODER TRAINING
# ============================================================================

def train_autoencoder(tiles_tensor, ae_config, device):
    """
    Trains the ViTAutoencoder.
    """
    print("\n--- Starting Autoencoder Training ---")
    print(f"Using device: {device}")
    
    dataset = TensorDataset(tiles_tensor)
    # Drop last batch if it's incomplete, simplifies training
    dataloader = DataLoader(dataset, batch_size=ae_config['batch_size'], shuffle=True, drop_last=True)
    
    num_batches = len(dataloader)
    if num_batches == 0:
        print("Error: Dataloader is empty. Check batch size and dataset.")
        return None
        
    print(f"Training on {len(tiles_tensor)} tiles, {num_batches} batches/epoch")

    # Initialize model
    autoencoder = ViTAutoencoder(
        tile_size=ae_config['tile_size'],
        patch_size=ae_config['patch_size'],
        embed_dim=ae_config['embed_dim'],
        depth=ae_config['depth'],
        num_heads=ae_config['num_heads'],
        mlp_ratio=ae_config['mlp_ratio'],
        decoder_hidden_dim=ae_config['decoder_hidden_dim']
    ).to(device)

    optimizer = optim.AdamW(autoencoder.parameters(), lr=ae_config['lr'])
    criterion = nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=ae_config['epochs'] * num_batches)

    # Calculate and print model size
    encoder_params = sum(p.numel() for p in autoencoder.encoder.parameters())
    
    # --- NEW: Calculate params for deep decoder ---
    decoder_params = sum(p.numel() for p in autoencoder.decoder_mlp.parameters())
    # --- END NEW ---
    
    print(f"Autoencoder parameters: {(encoder_params + decoder_params) / 1e6:.2f}M")
    print(f"  > Encoder (ViT):    {encoder_params / 1e6:.4f}M")
    print(f"  > Decoder (MLP):    {decoder_params / 1e6:.4f}M")

    start_time = time.time()
    
    # Outer epoch loop with persistent bar
    epoch_pbar = tqdm(range(ae_config['epochs']), desc="Training AE", unit="epoch")
    for epoch in epoch_pbar:
        autoencoder.train()
        total_loss = 0.0

        for (batch,) in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), GRAD_CLIP_VALUE)
            
            optimizer.step()
            scheduler.step() # Step scheduler every batch
            
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        
        # Update the persistent epoch progress bar
        epoch_pbar.set_postfix(avg_loss=f"{avg_loss:.6f}", lr=f"{current_lr:.2e}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    return autoencoder



# ============================================================================
#                           MAIN WORKFLOW
# ============================================================================

def main():
    """
    End-to-end workflow:
    1. Extract and tile GPT-2 MLP weights.
    2. Train the ViT-MLP autoencoder on the tiles.
    3. Reconstruct the weights and build a new GPT-2 model.
    4. Benchmark perplexity of original vs. reconstructed model.
    """
    print("--- GPT-2 ViT AutoEncoder Benchmark Script ---")
    
    # --- Config ---
    ae_config = {
        'tile_size': TILE_SIZE,
        'patch_size': PATCH_SIZE,
        'embed_dim': COMPRESSED_DIM,
        'depth': VIT_DEPTH,
        'num_heads': VIT_HEADS,
        'mlp_ratio': VIT_MLP_RATIO,
        'decoder_hidden_dim': DECODER_HIDDEN_DIM,
        'epochs': AE_EPOCHS,
        'batch_size': AE_BATCH_SIZE,
        'lr': AE_LEARNING_RATE,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Config: Model={MODEL_NAME}, Tile={TILE_SIZE}, Patch={PATCH_SIZE}, k={COMPRESSED_DIM}")
    print(f"Config: ViT_Depth={VIT_DEPTH}, ViT_Heads={VIT_HEADS}")
    print(f"Config: Decoder_Hidden={DECODER_HIDDEN_DIM}")
    print(f"Config: Epochs={AE_EPOCHS}, BatchSize={AE_BATCH_SIZE}, LR={AE_LEARNING_RATE}")

    # --- Step 1: Get Tiles ---
    tiles_tensor, tile_metadata = utils.get_gpt2_mlp_tiles(MODEL_NAME, TILE_SIZE)
    
    if tiles_tensor is None:
        print("Failed to get tiles. Exiting.")
        return

    # --- Step 2: Train Autoencoder ---
    autoencoder = train_autoencoder(tiles_tensor, ae_config, device)
    
    if autoencoder is None:
        print("Failed to train autoencoder. Exiting.")
        return
        
    # Free up tile tensor memory
    del tiles_tensor
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 3: Reconstruct Model ---
    reconstructed_model = eval.reconstruct_gpt2(autoencoder, MODEL_NAME, tile_metadata, device)
    
    if reconstructed_model is None:
        print("Failed to reconstruct model. Exiting.")
        return

    # Free up autoencoder memory
    del autoencoder
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- Step 4: Benchmark Perplexity ---
    print("\n--- Benchmarking Perplexity ---")
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    
    # Benchmark 1: Original Model
    #print("\nCalculating perplexity for ORIGINAL model...")
    #original_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    #original_ppl = calculate_perplexity(original_model, tokenizer, device)
    #del original_model # Free memory
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark 2: Reconstructed Model
    print("\nCalculating perplexity for RECONSTRUCTED model...")
    recon_ppl = eval.calculate_perplexity(reconstructed_model, tokenizer, device)
    del reconstructed_model # Free memory
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 5: Results ---
    print("\n--- FINAL RESULTS ---")
    #print(f"Original Perplexity:      {original_ppl:.4f}")
    print(f"Reconstructed Perplexity: {recon_ppl:.4f}")
    
    #if original_ppl > 0 and recon_ppl > 0:
        #increase = recon_ppl - original_ppl
        #percent_increase = (increase / original_ppl) * 100
        #print(f"Perplexity Increase:      {increase:+.4f} ({percent_increase:+.2f}%)")
    
    print("Benchmark complete.")

if __name__ == "__main__":
    main()

