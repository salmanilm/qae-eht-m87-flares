import torch
import torch.nn as nn
import torch.optim as optim

class CNNAutoencoder(nn.Module):
    """
    MINIMAL ARCHITECTURE for 4x4 patches:
    - Encoder: Compresses 1x4x4 → 64x1x1 (latent)
    - Decoder: Reconstructs 64x1x1 → 1x4x4
    
    WHY NO RESIDUAL/UPSAMPLE: On 4x4 patches, spatial dimension bugs.
    This simple architecture is trains faster and should be good.
    """
    
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        
        # --- ENCODER ---
        self.encoder = nn.Sequential(
            # 1x4x4 → 32x4x4
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32x4x4 → 32x2x2
            nn.MaxPool2d(2),
            
            # 32x2x2 → 64x2x2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x2x2 → latent_dimx1x1
            nn.Conv2d(64, latent_dim, kernel_size=2, stride=1, padding=0)
        )
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            # latent_dimx1x1 → 64x2x2
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x2x2 → 32x4x4 (stride=2 handles upsampling)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x4x4 → 1x4x4
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def get_reconstruction_error(self, x):
        reconstructed, _ = self.forward(x)
        mse = nn.functional.mse_loss(reconstructed, x, reduction='none')
        return mse.mean(dim=(1, 2, 3))


def create_optimizer(model, learning_rate: float = 0.001, weight_decay: float = 1e-4):
    return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_loss_function():
    return nn.MSELoss()


# quick test (ran this to verify shapes)
if __name__ == "__main__":
    model = CNNAutoencoder(latent_dim=64)
    dummy = torch.randn(1, 1, 4, 4)
    reconstructed, latent = model(dummy)
    
    print(f"Input: {dummy.shape}")
    print(f"Latent: {latent.shape}")
    print(f"Output: {reconstructed.shape}")
    print(f"Shapes match: {dummy.shape == reconstructed.shape}")