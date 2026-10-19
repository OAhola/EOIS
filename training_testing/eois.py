import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

# ---------- Functions for calculating feature losses for phase-locked and non-phase-locked components ----------

def morlet_cwt(x, freqs, fs, n_cycles):
    device = x.device
    dtype_r = x.dtype
    dtype_c = torch.complex64 if dtype_r == torch.float32 else torch.complex128

    freqs = torch.as_tensor(freqs, device=device, dtype=torch.float32)
    n_cycles = torch.as_tensor(n_cycles, device=device, dtype=torch.float32)

    n_trials, n_times = x.shape
    n_freqs = freqs.numel()

    # Precompute max padding once
    sigma_all = n_cycles / (2 * torch.pi * freqs)
    L_all = torch.ceil(3 * sigma_all * fs).to(torch.int64)
    L_max = int(L_all.max().item())
    dt = 1.0 / fs

    x_unsq = x.unsqueeze(1)                         # [N,1,T]
    x_pad = F.pad(x_unsq, (L_max, L_max), mode="reflect")  # [N,1,T+2L_max]

    # Build all wavelet kernels at the same length so they can be applied in one conv1d
    K_max = 2 * L_max + 1
    kernels = torch.zeros((2 * n_freqs, 1, K_max), dtype=dtype_r, device=device)

    # Build per-frequency wavelets
    for i in range(n_freqs):
        f = freqs[i]               # tensor scalar
        nc = n_cycles[i]           # tensor scalar
        sigma_t = nc / (2 * torch.pi * f)
        L = int(L_all[i].item())

        t = torch.arange(-L, L + 1, device=device, dtype=dtype_r) / fs
        t_c = t.to(dtype_c)

        # Morlet wavelet ψ(t) = exp(j 2π f t) * exp(-t^2 / (2 σ_t^2))
        carrier = torch.exp(1j * (2 * torch.pi * f.to(dtype_c)) * t_c)
        envelope = torch.exp(-(t_c.real**2) / (2 * (sigma_t.to(dtype_r) ** 2))).to(dtype_c)
        wavelet = carrier * envelope

        # Energy normalization
        norm = torch.sqrt((wavelet.abs() ** 2).sum() * dt)
        wavelet = wavelet / norm

        # Correlation with ψ* (no flip) for standard CWT
        w = torch.conj(wavelet)

        start = L_max - L
        kernels[2 * i,     0, start:start + 2 * L + 1] = w.real.to(dtype_r)
        kernels[2 * i + 1, 0, start:start + 2 * L + 1] = w.imag.to(dtype_r)

    # Convolution -> [N, 2F, T]
    conv = F.conv1d(x_pad, kernels)
    conv = conv.view(n_trials, n_freqs, 2, n_times)
    tfr = conv[:, :, 0, :].to(dtype_c) + 1j * conv[:, :, 1, :].to(dtype_c)
    
    return tfr



def ppc_mean(Zc, eps):
    """
    Zc: complex time-frequency representation values with shape [trials, ...]
    Returns: scalar PPC-based phase-locking measure (higher = more phase locking)
    """
    # Normalize to unit magnitude
    ph = Zc / (Zc.abs() + eps)          # [N, ...]
    N  = ph.size(0) #number of trials
    
    if N < 2:
        raise ValueError(f"Need at least 2 trials to calculate PPC, got {N} trials.")

    # sum across trials
    sum_ph = ph.sum(dim=0)              # [...]

    # PPC formula: (|Σ ph|^2 - N) / (N*(N-1))
    numer = (sum_ph.abs()**2 - N)
    denom = (N * (N - 1))
    ppc   = numer / denom               # [...]

    return ppc.mean()                   # scalar


def evoked_loss_pl(X, E, eps):
    evoked_target = X.mean(dim=0) # Average response across trials
    E_mean = E.mean(dim=0) # Average of phase-locked components
    
    # Evoked consistency: mean(E) ≈ evoked target (normalized squared error)
    loss_evoked_raw = (E_mean - evoked_target).pow(2).mean() # MSE
    evoked_target_power = evoked_target.pow(2).mean() # Normalization factor
    loss_evoked = loss_evoked_raw / (evoked_target_power + eps) # Loss of evoked reconstruction, normalized MSE
    
    return loss_evoked


def ongoing_induced_loss_npl(X_post, U_post, eps):
    """
    U_post: [B, T_post]
    X_post: [B, T_post]

    Penalize mean(U_post) ≈ 0.
    """
    # Mean over trials
    U_mean = U_post.mean(dim=0)          # [T_post]
    evoked = X_post.mean(dim=0)          # [T_post]
    loss = U_mean.pow(2).mean() # MSE
    loss = loss / (evoked.pow(2).mean() + eps) #normalized MSE

    return loss


def reco_loss(X, comp, eps):
    diff_power = (X - comp).pow(2).mean() # MSE across trials and times
    target_power = X.pow(2).mean() # Normalization factor
    loss_power = diff_power / (target_power + eps) # Loss of reconstruction, normalized MSE
    
    return loss_power


def component_power_loss(X, E, U, eps, win_len=None, hop=None):
    T = X.shape[1]

    # Default window: 20% of the trial
    if win_len is None:
        win_len = max(1, int(round(T / 5)))
    # Make odd for symmetric centering
    if win_len % 2 == 0:
        win_len += 1

    if hop is None:
        hop = max(1, win_len // 2)

    # Centered padding so every time contributes
    half = win_len // 2
    pad = (half, half)
    pad_mode = "reflect" if half < T else "replicate"

    Xp = F.pad(X.unsqueeze(1), pad, mode=pad_mode)
    Ep = F.pad(E.unsqueeze(1), pad, mode=pad_mode)
    Up = F.pad(U.unsqueeze(1), pad, mode=pad_mode)

    k = torch.ones(1, 1, win_len, device=X.device, dtype=X.dtype) / win_len

    px = F.conv1d(Xp.pow(2), k, stride=hop).squeeze(1)
    pe = F.conv1d(Ep.pow(2), k, stride=hop).squeeze(1)
    pu = F.conv1d(Up.pow(2), k, stride=hop).squeeze(1)

    valid = px > eps
    if not valid.any():
        return X.new_zeros(())

    ratio = (pe + pu) / px.clamp_min(eps)
    excess = torch.relu(ratio - 1.0)

    per_trial_sum = (excess * valid).sum(dim=1)
    per_trial_cnt = valid.sum(dim=1).clamp_min(1)
    return (per_trial_sum / per_trial_cnt).mean()

# ---------- Model architecture setup ----------

class DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout, use_gn=True):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = ((kernel_size - 1)//2) * dilation

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.norm1 = nn.GroupNorm(1, channels) if use_gn else nn.Identity()

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.norm2 = nn.GroupNorm(1, channels) if use_gn else nn.Identity()

    def forward(self, x):  # x: [N,C,T]
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return x + h


class EvokedOngoingInducedSeparator(nn.Module):
    def __init__(self, latent_dim=64, kernel_sizes=(5,5,7,9,9,9),
                 dilations=(1,2,4,8,16,32), dropout=0.1, use_gn=True, e_head_kernel=3):
        super().__init__()

        self.input_proj = nn.Conv1d(1, latent_dim, kernel_size=1)

        # dilated residual blocks
        self.blocks = nn.ModuleList([
            DilatedResBlock(latent_dim, k, d, dropout, use_gn)
            for d, k in zip(dilations, kernel_sizes)
        ])

        # evoked and ongoing/induced heads
        pad = (e_head_kernel - 1)//2
        self.E_head = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=e_head_kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(latent_dim, 1, kernel_size=1),
        )
        self.U_head = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(latent_dim, 1, kernel_size=1),
        )

        #initialization of the evoked head
        nn.init.zeros_(self.E_head[-1].weight)
        nn.init.zeros_(self.E_head[-1].bias)

    def forward(self, X):  # X: [N,T]
        z = self.input_proj(X.unsqueeze(1))  # [N,C,T]
        for blk in self.blocks:
            z = blk(z)
        E = self.E_head(z).squeeze(1)        # [N,T]
        U = self.U_head(z).squeeze(1)        # [N,T]
        return E, U


    def compute_loss(self, X, E, U, pre_lims, post_lims, fs, freqs, n_cycles, weights=None, gamma=1, eps=1e-8):
        """
        X: [B, T], E, U: [B, T]
        """
        # Default weights for features
        w = dict(
            trial_reco=3.0,     # Recovering the full trial by encouraging X = E + U.
            evoked_reco=8.0,    # Evoked consistency for E. The across-trial average of E should match the across-trial average of X.
            ongoing_induced_reco=3.0,   # Ongoing/Induced inconsistency for U. Should be zero when averaged across trials.
            e_ppc=3.0,        # High phase consistency for E across trials.
            u_ppc=6.0,        # Low phase consistency for U across trials.
            pre_reco=1.0,     # Match pre-stimulus X with pre-stimulus U.
            comp_power=1.0,   # Penalize cancelling components in E + U that show still in power
        )
        if weights is not None:
            allowed = set(w)
            # Assert that provided keys are a subset of allowed
            assert set(weights).issubset(allowed), (
                f"Unknown weight keys: {sorted(set(weights) - allowed)}. "
                f"Allowed keys: {sorted(allowed)}"
            )
            w.update(weights) #update dictionary keys if specific weights are provided

        # Open defined limits (in samples) of post and pre
        post_start, post_end = post_lims
        pre_start, pre_end = pre_lims
        X = X.detach() #detach the unsepareted data

        # --------------Calculate average evoked and ongoing/induced loss--------------
        loss_evoked = evoked_loss_pl(X[:, post_start:post_end], E[:, post_start:post_end], eps) # should match over trials
        loss_ongoing_induced = ongoing_induced_loss_npl(X[:, post_start:post_end], U[:, post_start:post_end], eps) # Should be near zero over trials.


        # --------------Calculate phase-related losses for both components--------------
        # complex Morlet wavelets (time-frequency representations, TFRs)
        Ec = morlet_cwt(E, freqs, fs, n_cycles)
        Uc = morlet_cwt(U, freqs, fs, n_cycles)
    
        # --------------PPC losses during post-stimulus--------------
        e_ppc = ppc_mean(Ec[:,:,post_start:post_end], eps)
        e_ppc_safe = torch.clamp(e_ppc, min=eps)
        loss_e_ppc = (1.0 - e_ppc_safe).pow(gamma) * (-torch.log(e_ppc_safe)) # Encourage high phase-locking for Ec_post
        

        loss_u_ppc = ppc_mean(Uc[:,:,post_start:post_end], eps) # Discourage high phase-locking for Uc_post

        #--------------Calculate the pre-stimulus loss--------------
        # Encourage that no phase-locked activity exists during the pre-stimulus period
        loss_u_pre = reco_loss(X[:,pre_start:pre_end], U[:,pre_start:pre_end], eps)

        #--------------Calculate trial recovery loss--------------
        loss_trial_reco = reco_loss(X, E + U, eps)

        #--------------Excess power loss to avoid cancelling artifacts--------------
        loss_comp_power = component_power_loss(X, E, U, eps)

        # total loss that guides the training as the weighted sum of individual losses
        total_loss = (
            w['trial_reco']       * loss_trial_reco
            + w['evoked_reco']    * loss_evoked
            + w['ongoing_induced_reco']   * loss_ongoing_induced
            + w['e_ppc']          * loss_e_ppc
            + w['u_ppc']          * loss_u_ppc
            + w['pre_reco']       * loss_u_pre
            + w['comp_power']     * loss_comp_power
        )

        #save specific losses (mainly for monitoring)
        loss_dict = {
            "trial_reco": float(loss_trial_reco.detach()),
            "evoked_mse": float(loss_evoked.detach()),
            "ongoing_induced_mse": float(loss_ongoing_induced.detach()),
            "e_ppc_loss": float(loss_e_ppc.detach()),
            "u_ppc_loss": float(loss_u_ppc.detach()),
            "pre_mse": float(loss_u_pre.detach()),
            "comp_power": float(loss_comp_power.detach()),
        }
        
        return total_loss, loss_dict




    
