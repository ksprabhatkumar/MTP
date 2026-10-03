import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDynamicGraph(nn.Module):
    """Computes the 3-part continuous probability adjacency matrix (MTP-2)."""
    def __init__(self, num_nodes, in_features, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Learnable scalars (alpha, beta, gamma)
        self.alpha = nn.Parameter(torch.ones(1) * 1.5) 
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)
        
        # A_learn (Global dataset kinematic priors)
        self.A_learn = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        
        # Projections for Instance-Specific Routing (A_dyn)
        self.q_proj = nn.Linear(in_features, hidden_dim)
        self.k_proj = nn.Linear(in_features, hidden_dim)
        self.temperature = hidden_dim ** 0.5 

    def forward(self, x, A_phys):
        # 🌟 FIXED: Changed F to F_dim so it doesn't overwrite torch.nn.functional
        B_T, N, F_dim = x.shape
        
        x_guarded = torch.tanh(x)
        
        Q = self.q_proj(x_guarded) 
        K = self.k_proj(x_guarded) 
        
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.temperature
        A_dyn = F.softmax(attention_scores, dim=-1) 
        
        A_final = (self.alpha * A_phys.unsqueeze(0)) + \
                  (self.beta * self.A_learn.unsqueeze(0)) + \
                  (self.gamma * A_dyn)
                  
        return A_final

class KinematicGraphEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )

    def forward(self, x, A_final):
        msg = torch.bmm(A_final, x)
        B_T, N, F_out = msg.shape
        
        msg_flat = msg.view(B_T * N, F_out)
        out_flat = self.mlp(msg_flat)
        return out_flat.view(B_T, N, -1)

class KinematicGraphDecoder(nn.Module):
    def __init__(self, hidden_dim, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, latent_x, A_final):
        msg = torch.bmm(A_final, latent_x)
        B_T, N, F_out = msg.shape
        
        msg_flat = msg.view(B_T * N, F_out)
        out_flat = self.mlp(msg_flat)
        return out_flat.view(B_T, N, -1)

class PurePyTorchMambaBlock(nn.Module):
    def __init__(self, d_model, expand=2, d_conv=4, dropout=0.3):
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, 
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.ssm_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        residual = x
        x = self.norm(x)
        
        x_proj = self.in_proj(x)
        x_proj, res = x_proj.chunk(2, dim=-1)
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        ssm_out = self.ssm_proj(x_conv) * F.silu(res)
        out = self.out_proj(ssm_out)
        out = self.dropout(out)
        
        return out + residual

class GAE_Mamba_WHAR(nn.Module):
    def __init__(self, num_nodes=5, in_features=9, hidden_dim=64, num_classes=19, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        if num_nodes == 5:
            edges = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [0, 4], [4, 0]]
        elif num_nodes == 9:
            edges = [
                [0, 1], [1, 0], [0, 2], [2, 0], [0, 5], [5, 0], [0, 6], [6, 0],
                [1, 3], [3, 1], [2, 4], [4, 2],
                [5, 7], [7, 5], [6, 8], [8, 6]
            ]
        else:
            raise ValueError("Unsupported number of nodes!")
            
        dense_A = torch.zeros((num_nodes, num_nodes))
        for i, j in edges:
            dense_A[i, j] = 1.0
        self.register_buffer('A_phys', dense_A)
        
        self.dynamic_graph = ResidualDynamicGraph(num_nodes, in_features, hidden_dim)
        
        self.encoder = KinematicGraphEncoder(in_features, hidden_dim)
        self.decoder = KinematicGraphDecoder(hidden_dim, in_features)
        
        mamba_dim = num_nodes * hidden_dim
        self.mamba = PurePyTorchMambaBlock(d_model=mamba_dim, dropout=dropout)
        
        self.norm_final = nn.LayerNorm(mamba_dim)
        self.classifier = nn.Linear(mamba_dim, num_classes)

    def forward(self, x_corrupt):
        B, T, N, F_dim = x_corrupt.shape
        x_flat_seq = x_corrupt.view(B * T, N, F_dim)
        
        A_final = self.dynamic_graph(x_flat_seq, self.A_phys)
        
        latent_graph = self.encoder(x_flat_seq, A_final)
        x_recon_flat = self.decoder(latent_graph, A_final)
        x_recon = x_recon_flat.view(B, T, N, F_dim)
        
        seq_x = latent_graph.view(B, T, N * self.hidden_dim)
        mamba_out = self.mamba(seq_x)
        
        pooled_state = mamba_out.mean(dim=1)
        final_state = self.norm_final(pooled_state)
        logits = self.classifier(final_state)
        
        return logits, x_recon