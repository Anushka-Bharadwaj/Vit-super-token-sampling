#Custom imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from tqdm import tqdm
import glob
import datetime
import json
import seaborn as sns
from contextlib import nullcontext
import matplotlib.pyplot as plt
import timm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif obj is None:
        return None
    return obj
# Metrics
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# Function to set the random seed for reproducibility

def compute_model_statistics(model, input_size=(3, 224, 224), device="cpu"):
    """
    Compute FLOPs, total parameters, estimated memory usage, and average inference time.
    Requires the 'ptflops' package.
    """
    try:
        from ptflops import get_model_complexity_info
        with torch.cuda.device(0) if device == "cuda" else nullcontext():
            flops, ptflops_params = get_model_complexity_info(
                model, input_size, as_strings=True,
                print_per_layer_stat=False, verbose=False
            )
    except ImportError:
        flops, ptflops_params = "N/A", "N/A"
        print("ptflops package not found. Skipping FLOPs calculation.")

    total_params = sum(p.numel() for p in model.parameters())
    estimated_memory_usage_bytes = total_params * 4  # assuming float32 (4 bytes per parameter)

    dummy_input = torch.randn(1, *input_size).to(device)
    model.eval()
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    # Measure inference time over 100 runs.
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 100

    stats = {
        "flops": flops,
        "ptflops_params": ptflops_params,
        "total_params": total_params,
        "estimated_memory_usage_bytes": estimated_memory_usage_bytes,
        "avg_inference_time_seconds": avg_inference_time
    }
    return stats


# Function to calculate various metrics for classification tasks
def calculate_metrics(y_true,y_pred,y_score=None):
    metrics={}
    metrics["accuracy_score"]=accuracy_score(y_pred,y_true)
    metrics["top_1_accuracy"]=metrics["accuracy_score"]
    if y_score is not None and y_score.shape[1]>=3:
        top3_correct=0
        for i,true_label in enumerate(y_true):
            top3_indices=np.argsort(y_score[i])[::-1][:3] 
            if true_label in top3_indices:
                top3_correct+=1
            metrics["top_3_accuracy"]=top3_correct/len(y_true)
            
    else:
        if y_score is not None and y_score.shape[1]<3:
            print("Less than 3 classes so top_3 accurcay will be same as top_1 accurcay")
            metrics["top_3_accuracy"]=metrics["top_1 accuracy"]
        else:
            metrics["top_3_accuracy"]=None
    # Precision
    metrics['precision_micro']=precision_score(y_true,y_pred,average='micro',zero_division=0)
    metrics['precision_macro']=precision_score(y_true,y_pred,average='macro',zero_division=0)
    metrics['precision_weighted']=precision_score(y_true,y_pred,average='weighted',zero_division=0)
    
    # Recall
    metrics['recall_micro']=recall_score(y_true,y_pred,average='micro',zero_division=0)
    metrics['recall_macro']=recall_score(y_true,y_pred,average='macro',zero_division=0)
    metrics['recall_weighted']=recall_score(y_true,y_pred,average='weighted',zero_division=0)
    
    # F1 Score
    metrics['f1_micro']=f1_score(y_true,y_pred,average='micro',zero_division=0)
    metrics['f1_macro']=f1_score(y_true,y_pred,average='macro',zero_division=0)
    metrics['f1_weighted']=f1_score(y_true,y_pred,average='weighted',zero_division=0)

    if y_score is not None:
        try:
            # One-hot encode the true labels for multi-class ROC AUC
            y_true_onehot = np.zeros((len(y_true),len(np.unique(y_true))))
            for i, val in enumerate(y_true):
                y_true_onehot[i,val]=1
                
            metrics['auc_micro']=roc_auc_score(y_true_onehot,y_score,average='micro',multi_class='ovr')
            metrics['auc_macro']=roc_auc_score(y_true_onehot,y_score,average='macro',multi_class='ovr')
            metrics['auc_weighted']=roc_auc_score(y_true_onehot,y_score,average='weighted',multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
            metrics['auc_micro']=metrics['auc_macro']=metrics['auc_weighted']=None

    return metrics


# Function to save model checkpoints
def find_latest_checkpoint(save_dir):
    checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint and return the starting epoch."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)  # <-- Add weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'] + 1  # Return next epoch to start from
    

#Data directory
dir="SoyMCData"

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


#Train and validation datasets
train_dataset=datasets.ImageFolder(os.path.join(dir,"train"),transform=transform)
test_dataset=datasets.ImageFolder(os.path.join(dir,"test"),transform=transform)
val_dataset=datasets.ImageFolder(os.path.join(dir,"val"),transform=transform)
num_classes=len(train_dataset.classes)
num_classes


# Create data loaders
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)


# --- 2. CNN Stem before Patch Embedding ---
class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)
    

# ===== Modified Patch Embedding with Conv Stem support =====
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.conv_stem = ConvStem(n_channels, d_model)
        reduced_size = (img_size[0] // 4, img_size[1] // 4)
        self.grid_size = (reduced_size[0] // patch_size[0], reduced_size[1] // patch_size[1])
        self.n_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.proj(x)  # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiScalePatchEmbedding(nn.Module):
    """
    Extracts patch embeddings at multiple scales and concatenates them.
    """
    def __init__(self, d_model, img_size, patch_sizes, n_channels):
        super().__init__()
        self.img_size = img_size
        self.patch_sizes = patch_sizes if isinstance(patch_sizes, (list, tuple)) else [patch_sizes]
        self.n_channels = n_channels
        self.d_model = d_model

        # For each scale, create a conv stem and a patch projection
        self.stems = nn.ModuleList([
            ConvStem(n_channels, d_model) for _ in self.patch_sizes
        ])
        self.projs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=ps, stride=ps) for ps in self.patch_sizes
        ])

        # Calculate total number of patches
        self.n_patches = 0
        self.grid_sizes = []
        for ps in self.patch_sizes:
            reduced_size = (img_size[0] // 4, img_size[1] // 4)
            grid_size = (reduced_size[0] // ps, reduced_size[1] // ps)
            self.grid_sizes.append(grid_size)
            self.n_patches += grid_size[0] * grid_size[1]

    def forward(self, x):
        patch_tokens = []
        for stem, proj in zip(self.stems, self.projs):
            x_stem = stem(x)
            x_proj = proj(x_stem)
            x_flat = x_proj.flatten(2).transpose(1, 2)  # (B, N, d_model)
            patch_tokens.append(x_flat)
        x_cat = torch.cat(patch_tokens, dim=1)  # (B, sum(N), d_model)
        return x_cat

# ===== Positional Embedding (use RoPE if enabled) =====
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim, use_rope=False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.use_rope = use_rope

        self.dim = dim
        self.max_len = max_len

        if not use_rope:
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        else:
            # Precompute RoPE frequencies
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)

    def apply_rope(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        pos = torch.arange(N, device=x.device, dtype=x.dtype)
        sinusoid_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        sin = sinusoid_inp.sin()[None, :, :]
        cos = sinusoid_inp.cos()[None, :, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rope = torch.empty_like(x)
        x_rope[..., 0::2] = x1 * cos - x2 * sin
        x_rope[..., 1::2] = x1 * sin + x2 * cos
        return x_rope

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        if self.use_rope:
            # Only apply RoPE to patch tokens, not cls token
            x_cls, x_patch = x[:, :1], x[:, 1:]
            x_patch = self.apply_rope(x_patch)
            x = torch.cat((x_cls, x_patch), dim=1)
        else:
            x = x + self.pe[:, :x.size(1), :]
        return x

#Attention Head
class AttentionHead(nn.Module):
    def __init__(self,d_model,head_size):
        super().__init__()
        self.head_size=head_size
        # calculating the query matrix
        self.query=nn.Linear(d_model,head_size)
        # calculating the key matrix        
        self.key=nn.Linear(d_model,head_size)
        # calculating the value matrix
        self.value=nn.Linear(d_model,head_size)

        
    def forward(self,x):
        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)

        # calculating the dot product of query and key transpose (QKt)
        attention=Q@K.transpose(-2,-1)

        # scaling the attention bu 1/root(d)
        attention=attention/(self.head_size ** 0.5)

        # applying softmax
        attention=torch.softmax(attention,dim=-1)

        # calculating dot product of attention and V 
        attention=attention@V
        
        return attention
        
    
#Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super().__init__()

        self.head_size=d_model//n_head

        # calculating the Wo matrix
        self.W_o =nn.Linear(d_model,d_model)

        # calculating all the heads by stacking multiple(n_head) Attention Head
        self.heads=nn.ModuleList([AttentionHead(d_model,self.head_size) for _ in range(n_head)])

        
    def forward(self,x):
        # concatinating all the heads
        out =torch.cat([head(x) for head in self.heads],dim=-1)

        # calculating multi head attention (Z dot proct W_o) where Z is concatination of all the heads
        out=self.W_o(out)
        
        return out
        
# ===== Relative Position Bias for Transformer Attention =====
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, window_size):
        super().__init__()
        self.window_size = window_size  # [H, W]
        Wh, Ww = window_size
        num_relative_positions = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        return self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1],
            self.window_size[0]*self.window_size[1], -1)



# ===== Transformer Block with Optional Conv-based MLP =====
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, r_mlp=4, use_conv_mlp=False, relative_position_bias=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_conv_mlp = use_conv_mlp
        self.relative_position_bias = relative_position_bias

        if use_conv_mlp:
            self.mlp = nn.Sequential(
                nn.Conv1d(d_model, d_model * r_mlp, 1),
                nn.GELU(),
                nn.Conv1d(d_model * r_mlp, d_model, 1)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * r_mlp),
                nn.GELU(),
                nn.Linear(d_model * r_mlp, d_model)
            )

    def forward(self, x):
        x_norm = self.ln1(x)
        if self.relative_position_bias is not None:
            bias = self.relative_position_bias().permute(2, 0, 1)  # [num_heads, N, N]
            attn_output, _ = self.mha(x_norm, x_norm, x_norm, attn_mask=None, attn_bias=bias)
        else:
            attn_output, _ = self.mha(x_norm, x_norm, x_norm)

        x = x + attn_output
        x_norm = self.ln2(x)
        if self.use_conv_mlp:
            x = x + self.mlp(x_norm.transpose(1, 2)).transpose(1, 2)
        else:
            x = x + self.mlp(x_norm)
        return x

     

# Super Token Sampling
def super_token_sampling(x, k=50):
    """
    x: token embeddings (B, N, D)
    Keep top-k tokens based on L2 norm
    """
    cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]
    num_patch_tokens = patch_tokens.shape[1]
    k = min(k, num_patch_tokens)  # Ensure k does not exceed available tokens
    scores = patch_tokens.norm(dim=-1)  # (B, N-1)
    topk_indices = scores.topk(k, dim=1).indices  # (B, k)
    
    batch_size = x.size(0)
    selected_tokens = []
    for b in range(batch_size):
        selected = patch_tokens[b][topk_indices[b]]  # (k, D)
        selected_tokens.append(selected)
    selected_tokens = torch.stack(selected_tokens)  # (B, k, D)
    
    return torch.cat([cls_token, selected_tokens], dim=1)  # (B, k+1, D)
# ...existing code...
# --- Vision Transformer with Super Token Sampling ---
class ViT_STS(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=768, depth=12, heads=12, mlp_dim=3072, num_classes=1000, k=50, use_rope=False, use_conv_mlp=False):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.k = k
        self.patch_embed = PatchEmbedding(emb_dim, img_size, patch_size, 3)
        self.pos_embed = PositionalEmbedding(self.patch_embed.n_patches + 1, emb_dim, use_rope=use_rope)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.transformer = nn.Sequential(*[
            TransformerEncoder(emb_dim, heads, r_mlp=mlp_dim//emb_dim, use_conv_mlp=use_conv_mlp)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x)
        x = self.transformer(x)
        x = super_token_sampling(x, self.k)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# Define hyperparameters
d_model = 64
n_classes = 4
img_size = (224, 224)
patch_size = (16, 16)
n_heads = 4
n_layers = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Vision Transformer with Super Token Sampling
model = ViT_STS(
    img_size=img_size[0],
    patch_size=patch_size[0],
    emb_dim=d_model,
    depth=n_layers,
    heads=n_heads,
    mlp_dim=d_model*4,
    num_classes=n_classes,
    k=50,
    use_rope=True,
    use_conv_mlp=True

)

# Function to train and evaluate the model
def train_eval(model,lr=1e-4,epochs=1,save_dir="./results"):
    # create dir if not exsists
    os.makedirs(save_dir,exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    #add learning rate scheduler
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5)
    
    train_losses, val_losses = [], []
    train_metrics_history = [] 
    val_metrics_history = []
    best_metrics = {
        'val_loss': float('inf'),
        'val_top1': 0.0,
        'val_top3': 0.0,
        'epoch': 0
    }
   
    #try to load the latest checkpoint
    latest_checkpoint=find_latest_checkpoint(save_dir)
    start_epoch=0
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
        print(f"Resuming training from epoch {start_epoch}")
        
        # Load metrics history if available
        metrics_file = os.path.join(save_dir, "results.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                results = json.load(f)
                train_losses = results.get("train_losses", [])
                val_losses = results.get("val_losses", [])
                train_metrics_history = results.get("train_metrics_history", [])
                val_metrics_history = results.get("val_metrics_history", [])
                
                # Load best metrics from history
                if val_metrics_history:
                    best_epoch_idx = min(range(len(val_metrics_history)), 
                                       key=lambda i: val_metrics_history[i].get("loss", float('inf')))
                    best_metrics = {
                        'val_loss': val_metrics_history[best_epoch_idx].get("loss", float('inf')),
                        'val_top1': val_metrics_history[best_epoch_idx].get("top1_accuracy", 0.0),
                        'val_top3': val_metrics_history[best_epoch_idx].get("top3_accuracy", 0.0),
                        'epoch': best_epoch_idx
                    }
                print(f"Loaded metrics history from previous training")
    
    
    training_start_time=time.time()
    
    for epoch in range(start_epoch,epochs):
        model.train()
        running_loss=0
        epoch_start_time=time.time()
        train_y_true, train_y_pred, train_y_score = [], [], []
        for inputs,labels in tqdm(train_loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            output=model(inputs)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            _,predicted=torch.max(output,1)
            train_y_true.extend(labels.cpu().numpy())
            train_y_pred.extend(predicted.cpu().numpy())
            train_y_score.extend(torch.softmax(output, dim=1).detach().cpu().numpy())
            
        training_loss=running_loss/len(train_loader)
        train_losses.append(training_loss)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(train_y_true, train_y_pred, np.array(train_y_score))
        train_metrics['loss'] = training_loss  # Add loss to the metrics dictionary
        train_metrics['y_true'] = train_y_true  # <-- Add this
        train_metrics['y_pred'] = train_y_pred  # <-- Add this
        train_metrics_history.append(train_metrics)

        model.eval()
        running_loss=0
        val_y_true, val_y_pred, val_y_score = [], [], []
        with torch.inference_mode():
            for inputs,labels in tqdm(val_loader):
                inputs=inputs.to(device)
                labels=labels.to(device)
                output=model(inputs)
                loss=criterion(output,labels)
                running_loss+=loss.item()
                _,predicted=torch.max(output,1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(predicted.cpu().numpy())
                val_y_score.extend(torch.softmax(output, dim=1).detach().cpu().numpy())
                
        val_loss=running_loss/len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_metrics=calculate_metrics(val_y_true,val_y_pred,np.array(val_y_score))
        val_metrics['loss'] = val_loss  # Add loss to the metrics dictionary
        val_metrics_history.append(val_metrics)


        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {training_loss:.4f}")
        print(f"  Train Top-1: {train_metrics['top_1_accuracy']*100:.2f}%")
        print(f"  Train Top-3: {train_metrics['top_3_accuracy']*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Top-1: {val_metrics['top_1_accuracy']*100:.2f}%")
        print(f"  Val Top-3: {val_metrics['top_3_accuracy']*100:.2f}%")

        scheduler.step(val_loss)

        if val_loss<best_metrics['val_loss']:
            best_metrics['val_loss']=val_loss
            best_metrics['val_top1']=val_metrics['top_1_accuracy']
            best_metrics['val_top3']=val_metrics['top_3_accuracy']
            checkpoint_path=os.path.join(save_dir,"best_model_checkpoint.pth")
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'loss':val_loss,
                'metrics':val_metrics,                
            },checkpoint_path)
            print(f"  Regular checkpoint saved for epoch {epoch+1}")
            
        if (epoch+1)%10==0:
            checkpoint_path=os.path.join(save_dir,f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'loss':val_loss,
                'metrics':val_metrics
            },checkpoint_path)
            print(f"Reguler Checkpoint saved for epoch {epoch+1}")
        # Save intermediate results after each epoch
        results = {
            "train_losses": [m['loss'] for m in train_metrics_history],
            "val_losses": [m['loss'] for m in val_metrics_history],
            "train_metrics_history": train_metrics_history,
            "val_metrics_history": val_metrics_history,
            "best_validation": {
                "epoch": best_metrics['epoch'] + 1,
                "metrics": val_metrics_history[best_metrics['epoch']] if (val_metrics_history and best_metrics['epoch'] < len(val_metrics_history)) else (val_metrics_history[-1] if val_metrics_history else {})
            }
        }
        results_path = os.path.join(save_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(convert_to_serializable(results), f, indent=4)    

    # Calculate total training time
    total_training_time=time.time()-training_start_time
    training_time_formatted = str(datetime.timedelta(seconds=int(total_training_time)))
    print(f"\nTotal training time: {training_time_formatted}")

    #Testing loop
    model.eval()
    test_loss=0
    test_y_true,test_y_pred,test_y_score=[],[],[]
    with torch.inference_mode():
        for inputs,labels in tqdm(test_loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            output=model(inputs)
            loss=criterion(output,labels)
            test_loss+=loss.item()
            _,predicted=torch.max(output,1)
            test_y_true.extend(labels.cpu().numpy())
            test_y_pred.extend(predicted.cpu().numpy())
            test_y_score.extend(torch.softmax(output,dim=1).detach().cpu().numpy())
    test_loss/=len(test_loader)
    test_metrics=calculate_metrics(test_y_true,test_y_pred,np.array(test_y_score))

    # Print final test results
    print("\nFinal Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Top-1: {test_metrics['top_1_accuracy']*100:.2f}%")
    print(f"  Top-3: {test_metrics['top_3_accuracy']*100:.2f}%")

    # Save all metrics history
    metrics_history = {
        'train': train_metrics_history,
        'val': val_metrics_history,
        'test': test_metrics,
        'best_validation': {
            'epoch': best_metrics['epoch'],
            'metrics': val_metrics_history[best_metrics['epoch']] if (val_metrics_history and best_metrics['epoch'] < len(val_metrics_history)) else (val_metrics_history[-1] if val_metrics_history else {})
        }
    }

    #plot comparison btw top_1_acc and top_3_acc
    plt.figure(figsize=(10,6))
    # Plot comparison of top-1 and top-3 accuracies
    plt.figure(figsize=(10, 6))
    # Use the actual length of history instead of num_epochs
    actual_epochs = len(train_metrics_history)
    epoch_range = range(1, actual_epochs + 1)
    train_top1 = [m['top_1_accuracy'] * 100 for m in train_metrics_history]
    train_top3 = [m['top_3_accuracy'] * 100 for m in train_metrics_history]
    val_top1 = [m['top_1_accuracy'] * 100 for m in val_metrics_history]
    val_top3 = [m['top_3_accuracy'] * 100 for m in val_metrics_history]
    plt.plot(epoch_range, train_top1, 'b-', label='Train Top-1')
    plt.plot(epoch_range, train_top3, 'b--', label='Train Top-3')
    plt.plot(epoch_range, val_top1, 'r-', label='Val Top-1')
    plt.plot(epoch_range, val_top3, 'r--', label='Val Top-3')
    plt.title('Top-1 and Top-3 Accuracies Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'top1_top3_comparison.png'))
    plt.close()

    #plot loss curves
    plt.figure(figsize=(10,6))
    train_losses = [epoch_data['loss'] for epoch_data in train_metrics_history]
    val_losses = [epoch_data['loss'] for epoch_data in val_metrics_history]  
    plt.plot(epoch_range,train_losses,'b-',label='Train Loss')
    plt.plot(epoch_range,val_losses,'r-',label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir,'loss_curves.png'))
    plt.close()

    # Confusion matrix plot for test data
    conf_matrix = confusion_matrix(test_y_true, test_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    conf_matrix_path=os.path.join(save_dir, "test_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    
    # Confusion matrix plot for training data
    if train_metrics_history:
        train_conf_matrix = confusion_matrix(train_metrics_history[-1]['y_true'], train_metrics_history[-1]['y_pred'])
        plt.figure(figsize=(10,8))
        sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Training Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        train_conf_matrix_path = os.path.join(save_dir, "train_conf_matrix.png")
        plt.savefig(train_conf_matrix_path)
        plt.close()
    else:
        print("Warning: No training metrics available to plot training confusion matrix.")
        train_conf_matrix_path = None

    # Save classification Reports
    train_y_true_all = []
    train_y_pred_all = []
    for epoch in range(len(train_metrics_history)):
        train_y_true_all.extend(train_metrics_history[epoch].get('y_true', []))
        train_y_pred_all.extend(train_metrics_history[epoch].get('y_pred', []))
    
    # If we don't have the raw predictions stored in metrics history, use the last epoch's data
    if not train_y_true_all:
        train_cls_report=classification_report(train_y_true,train_y_pred)
    else:
        train_cls_report=classification_report(train_y_true_all,train_y_pred_all)
    
    test_cls_report=classification_report(test_y_true,test_y_pred)
    
    train_report_path = os.path.join(save_dir,"train_classification_report.txt")
    with open(train_report_path,"w") as f:
        f.write(train_cls_report)
    
    test_report_path=os.path.join(save_dir,"test_classification_report.txt")
    with open(test_report_path,"w") as f:
        f.write(test_cls_report)

    # Save detailed metrics
    metrics_path = os.path.join(save_dir, "detailed_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("TRAINING METRICS (Final Epoch):\n")
        f.write("=============================\n")
        for metric, value in train_metrics_history[-1].items():
            if value is None:
                f.write(f"{metric}: N/A\n")
            elif isinstance(value, (float, int, np.floating, np.integer)):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {str(value)}\n")
        
        f.write("\nVALIDATION METRICS (Best Epoch):\n")
        f.write("==============================\n")
        best_val_metrics = val_metrics_history[best_metrics['epoch']] if (val_metrics_history and best_metrics['epoch'] < len(val_metrics_history)) else (val_metrics_history[-1] if val_metrics_history else {})
        for metric, value in best_val_metrics.items():
            if value is not None:
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: N/A\n")
        
        f.write("\nTEST METRICS:\n")
        f.write("=============\n")
        for metric, value in test_metrics.items():
            if value is not None:
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: N/A\n")
        
        # Add training time information
        f.write("\nTRAINING TIME:\n")
        f.write("=============\n")
        f.write(f"Total training time: {training_time_formatted}\n")
        f.write(f"Average time per epoch: {total_training_time/epochs:.2f} seconds\n")

    # Save epoch wise data
    epoch_metric_path=os.path.join(save_dir,"training_metrics.txt")
    with open(epoch_metric_path,"w") as f:
        f.write("Epoch wise Training and Validation Metrics\n")
        for i, epoch_idx in enumerate(range(len(train_metrics_history))):
            actual_epoch = start_epoch + i  # Calculate the true epoch number
            f.write(f"Epoch {actual_epoch+1}:\n")
            f.write(f"  Train Loss: {train_metrics_history[epoch_idx]['loss']:.4f}\n")
            f.write(f"  Train Top-1: {train_metrics_history[epoch_idx]['top_1_accuracy']*100:.2f}%\n")
            f.write(f"  Train Top-3: {train_metrics_history[epoch_idx]['top_3_accuracy']*100:.2f}%\n")
            f.write(f"  Val Loss: {val_metrics_history[epoch_idx]['loss']:.4f}\n")
            f.write(f"  Val Top-1: {val_metrics_history[epoch_idx]['top_1_accuracy']*100:.2f}%\n")
            f.write(f"  Val Top-3: {val_metrics_history[epoch_idx]['top_3_accuracy']*100:.2f}%\n")
            f.write("\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Top-1: {test_metrics['top_1_accuracy']*100:.2f}%\n")
        f.write(f"Test Top-3: {test_metrics['top_3_accuracy']*100:.2f}%\n")
        f.write(f"Total training time: {training_time_formatted}\n")

    # Compute and Save model Statistics
    model_stats = compute_model_statistics(model, input_size=(3, 224, 224), device=device)
    stats_path = os.path.join(save_dir, "model_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(convert_to_serializable(model_stats), f, indent=4)

    # Compile all results in a dictionary and save to JSON
    results = {
        "train_losses": [m['loss'] for m in train_metrics_history],
        "val_losses": [m['loss'] for m in val_metrics_history],
        "train_top1_accuracies": train_top1,
        "train_top3_accuracies": train_top3,
        "val_top1_accuracies": val_top1,
        "val_top3_accuracies": val_top3,
        "test_loss": test_loss,
        "test_metrics": convert_to_serializable(test_metrics),
        "best_validation": {
            "epoch": best_metrics['epoch'] + 1,
            "metrics": val_metrics_history[best_metrics['epoch']] if (val_metrics_history and best_metrics['epoch'] < len(val_metrics_history)) else (val_metrics_history[-1] if val_metrics_history else {})
        },
        "model_statistics": convert_to_serializable(model_stats),
        "training_time": {
            "total_seconds": total_training_time,
            "formatted": training_time_formatted,
            "average_epoch_seconds": total_training_time/epochs
        },
        "plots": {
            "top1_top3_comparison": os.path.join(save_dir, 'top1_top3_comparison.png'),
            "loss_curves": os.path.join(save_dir, 'loss_curves.png'),
            "test_confusion_matrix": conf_matrix_path,
            "train_confusion_matrix": train_conf_matrix_path
        }
    }
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=4)

    # Save the trained model
    model_save_path = os.path.join(save_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
    print(f"All outputs have been saved to {os.path.abspath(save_dir)}")

    return model, metrics_history   



#Set the learning rate, epochs and save directory
# Hyperparameters
lr = 0.005
epochs = 5
save_dir=r"C:\\Users\\Hp\\CascadeProjects\\vision_transformer\\STS_vit"
train_eval(model,lr,epochs,save_dir)