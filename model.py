import torch
from torchvision.models import mobilenet_v3_large
from torch import nn

class TokenPooling(nn.Module):
    def __init__(self, keep_tokens: int, use_weighted: bool = True):
        super().__init__()
        self.keep_tokens = keep_tokens
        self.use_weighted = use_weighted

    def forward(self, x: torch.Tensor, significance: torch.Tensor = None) -> torch.Tensor:
        B, N_plus_1, D = x.shape
        # pisahkan cls_token dan token biasa
        cls_token, tokens = x[:, :1, :], x[:, 1:, :]  # (B, 1, D), (B, N, D)

        if self.keep_tokens >= tokens.shape[1]:
            return x  # tidak perlu pooling

        if not self.use_weighted:
            significance = torch.ones(tokens.shape[:2], device=x.device)

        # Ambil top-k indeks token berdasarkan skor
        topk_scores, topk_indices = torch.topk(significance, self.keep_tokens, dim=1)  # (B, K)

        # Ambil token berdasarkan indeks top-k
        B_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self.keep_tokens)  # (B, K)
        pooled_tokens = tokens[B_idx, topk_indices]  # (B, K, D)

        return torch.cat([cls_token, pooled_tokens], dim=1)  # (B, K+1, D)

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0.):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
        self.attn_weights = None
    def forward(self, x):
        attn_output, attn_weights = self.multihead_attn(query=x,
                                             key=x,
                                             value=x,
                                             need_weights=True,
                                             average_attn_weights=False)
        self.attn_weights = attn_weights
        return attn_output, attn_weights
    
class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 dropout:float=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    def forward(self, x):
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.,
                 attn_dropout:float=0.):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)
        
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)
        
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
    def forward(self, x):
        x_attn, attn_weights = self.msa_block(self.layer_norm1(x))
        x = x_attn + x
        
        x = self.mlp_block(self.layer_norm2(x)) + x 
        
        return x, attn_weights

class ViTMobilenet(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 img_size:int=224,
                 num_transformer_layers:int=12,
                 embedding_dim:int=768, 
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0., 
                 mlp_dropout:float=0.,
                 embedding_dropout:float=0., 
                 num_classes:int=1000):
        super().__init__()
        
        # memastikan ukuran gambar dapat diproses
        assert img_size % 32 == 0, f"Image size must be divisible by 32, image size: {img_size}"
        
        # inisiasai mobilenetv3 sebagai backbone
        self.mobilenet = mobilenet_v3_large(pretrained=True).features
        
        # proyeksi channel agar dapat diproses oleh encoder
        self.projection = nn.Conv2d(in_channels=960, 
                                    out_channels=embedding_dim,
                                    kernel_size=1)
        
        # inisiasi token CLS
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # jumlah patch
        self.num_patches = (img_size // 32) ** 2 
        
        # inisiasi position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        
        # inisiasi dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # inisiasi transformer encoder
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout,
                                                                            attn_dropout=attn_dropout) for _ in range(num_transformer_layers)])
        
        # inisiasi untuk token downsampling
        self.keep_tokens = [49, 35, 35, 35, 26, 26, 20, 20, 20, 12, 12, 12] # parameter retention
        # Tambahkan Token Downsampling per layer (jumlah token disesuaikan)
        self.token_pools = nn.ModuleList([
            TokenPooling(keep_tokens=k, use_weighted=True) for k in self.keep_tokens
        ])

        # inisiasi classification head
        self.norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)
        self.head = nn.Linear(in_features=embedding_dim, out_features=num_classes)
    
    def forward(self, pixel_values, labels=None):
        # ambil batch
        batch_size = pixel_values.shape[0]

        # ekstraksi fitur menggunakan mobilenetv3
        features = self.mobilenet(pixel_values)  # Output shape: (batch_size, 1280, H', W')
        features = self.projection(features)  # Project ke embedding_dim: (batch_size, embedding_dim, H', W')

        # Flatten fitur map ke  sequence of tokens
        features = features.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_dim)
        
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        # tambahkan token CLS
        x = torch.cat((class_token, features), dim=1)  # Shape: (batch_size, num_patches + 1, embedding_dim)

        x = x + self.position_embedding

        x = self.embedding_dropout(x)
        
        significance_scores = []

        for i, block in enumerate(self.transformer_encoder):
            x, attn_weights = block(x) # SHape: (B, num_heads, num_tokens, num_tokens)
            
            # Hitung significance score: total attention yang diterima setiap token
            score = attn_weights.sum(dim=1).sum(dim=1)[:, 1:]  # shape: (B, N-1)
            
            significance_scores.append(score)
            
            if self.token_pools[i].keep_tokens > 0:
                x = self.token_pools[i](x, significance=score)
            else:
                x = x[:, :1, :]  # hanya CLS token

        x = self.norm(x)
        
        cls_token_final = x[:, 0]

        logits = self.head(cls_token_final)

        return logits