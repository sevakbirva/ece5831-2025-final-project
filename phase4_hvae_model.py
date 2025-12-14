"""
Phase 4: Hierarchical VAE Model for Multi-Task Trip Intent Learning

This module defines a Hierarchical VAE that:
- Embeds categorical features (stations, bike type, member/casual)
- Consumes numeric context features (duration norm, lat/lng norm, time features)
- Learns two latent spaces:
    - z_global: shared intent (e.g., commute vs leisure vs tourism)
    - z_individual: residual / individual variation
- Decodes into multiple tasks:
    - Task 1: Duration (log1p) with uncertainty (Gaussian NLL)
    - Task 2: Station demand contribution (start_station_day_share) regression
    - Task 3: Rideable type classification (rideable_type_idx)
- Provides a compute_loss() method combining reconstruction + KL terms.

"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F


# -------------------------------------------------------------------
# Configuration dataclass
# -------------------------------------------------------------------

@dataclass
class HVAEConfig:
    # Category sizes
    num_start_stations: int
    num_end_stations: int
    num_ride_types: int
    num_member_types: int

    # Embedding dimensions
    emb_dim_station: int = 32
    emb_dim_ride_type: int = 8
    emb_dim_member: int = 4

    # Numeric feature dimension
    num_numeric_features: int = 10

    # Latent dimensions
    latent_dim_global: int = 16
    latent_dim_individual: int = 16

    # Encoder / decoder hidden sizes
    encoder_hidden_dim: int = 128
    decoder_hidden_dim: int = 128

    # Loss weights
    w_duration: float = 1.0
    w_demand: float = 1.0
    w_ride_type: float = 1.0
    beta_global: float = 1.0
    beta_individual: float = 1.0


# -------------------------------------------------------------------
# Utility: reparameterization trick and KL
# -------------------------------------------------------------------

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick:
        z = mu + std * eps, where eps ~ N(0, I)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(mu, diag(exp(logvar))) and N(0, I).

    Clamp mu and logvar for numerical stability to avoid huge KL explosions.
    Returns mean KL per sample: shape []
    """
    # Clamp for stability
    mu_clamped = torch.clamp(mu, -10.0, 10.0)
    logvar_clamped = torch.clamp(logvar, -10.0, 10.0)

    kl_per_sample = -0.5 * torch.sum(
        1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp(),
        dim=-1,
    )
    return kl_per_sample.mean()



# -------------------------------------------------------------------
# Hierarchical VAE
# -------------------------------------------------------------------

class HierarchicalVAE(nn.Module):
    def __init__(self, config: HVAEConfig):
        super().__init__()
        self.config = config

        # -------------------------
        # Embeddings for categorical inputs
        # -------------------------
        self.emb_start_station = nn.Embedding(config.num_start_stations, config.emb_dim_station)
        self.emb_end_station   = nn.Embedding(config.num_end_stations,   config.emb_dim_station)
        self.emb_member        = nn.Embedding(config.num_member_types,   config.emb_dim_member)

        cat_emb_total_dim = (
            2 * config.emb_dim_station
            + config.emb_dim_member
        )

        encoder_input_dim = cat_emb_total_dim + config.num_numeric_features

        # -------------------------
        # Shared encoder
        # -------------------------
        self.encoder_mlp = nn.Sequential(
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.ReLU(),
        )

        # Global latent parameters
        self.fc_mu_global = nn.Linear(config.encoder_hidden_dim, config.latent_dim_global)
        self.fc_logvar_global = nn.Linear(config.encoder_hidden_dim, config.latent_dim_global)

        # Individual latent parameters (condition on h + z_global)
        self.fc_individual_prep = nn.Linear(
            config.encoder_hidden_dim + config.latent_dim_global,
            config.encoder_hidden_dim,
        )
        self.fc_mu_individual = nn.Linear(
            config.encoder_hidden_dim,
            config.latent_dim_individual,
        )
        self.fc_logvar_individual = nn.Linear(
            config.encoder_hidden_dim,
            config.latent_dim_individual,
        )

        # -------------------------
        # Decoder: shared backbone
        # -------------------------
        decoder_input_dim = config.latent_dim_global + config.latent_dim_individual

        self.decoder_mlp = nn.Sequential(
            nn.Linear(decoder_input_dim, config.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.ReLU(),
        )

        # -------------------------
        # Task-specific heads
        # -------------------------

        # Task 1: Duration (log1p) heteroscedastic Gaussian regression
        self.head_duration = nn.Linear(config.decoder_hidden_dim, 2)
        # output[... , 0] -> mu, output[... , 1] -> logvar

        # Task 2: Demand share regression (start_station_day_share)
        self.head_demand = nn.Linear(config.decoder_hidden_dim, 1)

        # Task 3: Rideable type classification
        self.head_ride_type = nn.Linear(config.decoder_hidden_dim, config.num_ride_types)

    # ------------------------------------------------------------------
    # Encoding and decoding
    # ------------------------------------------------------------------

    def encode(self, x_cat: torch.Tensor, x_num: torch.Tensor):
        """
        x_cat: LongTensor [B, 3] with columns:
            0: start_station_id_idx
            1: end_station_id_idx
            2: member_casual_idx
        """
        # Unpack categorical indices
        start_idx  = x_cat[:, 0]
        end_idx    = x_cat[:, 1]
        member_idx = x_cat[:, 2]

        # Embeddings
        emb_start  = self.emb_start_station(start_idx)
        emb_end    = self.emb_end_station(end_idx)
        emb_member = self.emb_member(member_idx)

        # Concatenate embeddings + numeric features
        x_emb = torch.cat([emb_start, emb_end, emb_member], dim=-1)
        encoder_input = torch.cat([x_emb, x_num], dim=-1)

        # Shared encoder
        h = self.encoder_mlp(encoder_input)

        # Global latent
        mu_global = self.fc_mu_global(h)
        logvar_global = self.fc_logvar_global(h)
        z_global = reparameterize(mu_global, logvar_global)

        # Individual latent, conditioned on h + z_global
        h_ind_input = torch.cat([h, z_global], dim=-1)
        h_ind = torch.relu(self.fc_individual_prep(h_ind_input))
        mu_ind = self.fc_mu_individual(h_ind)
        logvar_ind = self.fc_logvar_individual(h_ind)
        z_ind = reparameterize(mu_ind, logvar_ind)

        return z_global, z_ind, mu_global, logvar_global, mu_ind, logvar_ind

    def decode(self, z_global: torch.Tensor, z_individual: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent variables into task-specific outputs.

        Returns:
            {
                "duration_mu": FloatTensor [B],
                "duration_logvar": FloatTensor [B],
                "demand_pred": FloatTensor [B],
                "ride_logits": FloatTensor [B, num_ride_types],
            }
        """
        z = torch.cat([z_global, z_individual], dim=-1)
        d = self.decoder_mlp(z)

        # Duration
        dur_out = self.head_duration(d)  # [B, 2]
        duration_mu = dur_out[:, 0]
        duration_logvar = dur_out[:, 1]

        # Demand
        demand_pred = self.head_demand(d).squeeze(-1)  # [B]

        # Ride type logits
        ride_logits = self.head_ride_type(d)  # [B, num_ride_types]

        return {
            "duration_mu": duration_mu,
            "duration_logvar": duration_logvar,
            "demand_pred": demand_pred,
            "ride_logits": ride_logits,
        }

    def forward(
        self, x_cat: torch.Tensor, x_num: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Full forward pass:
            - encode -> z_global, z_individual
            - decode -> task outputs

        Returns:
            outputs: dict with decoded predictions
            latents: dict with z's and parameters
        """
        z_global, z_ind, mu_g, logvar_g, mu_i, logvar_i = self.encode(x_cat, x_num)
        outputs = self.decode(z_global, z_ind)

        latents = {
            "z_global": z_global,
            "z_individual": z_ind,
            "mu_global": mu_g,
            "logvar_global": logvar_g,
            "mu_individual": mu_i,
            "logvar_individual": logvar_i,
        }

        return outputs, latents

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _duration_nll(
        self,
        pred_mu: torch.Tensor,
        pred_logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood for duration (log1p) with stability clamps.

        NLL = 0.5 * [ log(2π) + logσ² + (y - μ)² / σ² ]

        Returns mean NLL over batch.
        """
        # Ensure shapes align
        target = target.view_as(pred_mu)

        # Clamp for numerical stability
        # - mu within a reasonable range of log1p(duration)
        # - logvar so that exp(logvar) stays in a safe range
        pred_mu_clamped = torch.clamp(pred_mu, -10.0, 10.0)
        logvar_clamped = torch.clamp(pred_logvar, -5.0, 5.0)  # σ² in [e^-5, e^5]

        var = torch.exp(logvar_clamped)
        var = torch.clamp(var, min=1e-6)

        nll = 0.5 * (
            math.log(2 * math.pi)
            + logvar_clamped
            + (target - pred_mu_clamped).pow(2) / var
        )
        return nll.mean()


    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full HVAE loss and its components.

        Args:
            batch: dict with keys:
                "x_cat": LongTensor [B, 4]
                "x_num": FloatTensor [B, D_num]
                "targets": dict with:
                    "duration_log1p": FloatTensor [B]
                    "demand_share":  FloatTensor [B]
                    "rideable_type": LongTensor [B]
            loss_weights: overrides for default config weights (optional).

        Returns:
            dict with:
                "loss": total scalar loss
                "loss_duration": duration reconstruction loss
                "loss_demand": demand regression loss
                "loss_ride_type": ride type classification loss
                "kl_global": KL(z_global || N(0,I))
                "kl_individual": KL(z_individual || N(0,I))
        """
        if loss_weights is None:
            loss_weights = {
                "w_duration": self.config.w_duration,
                "w_demand": self.config.w_demand,
                "w_ride_type": self.config.w_ride_type,
                "beta_global": self.config.beta_global,
                "beta_individual": self.config.beta_individual,
            }

        x_cat = batch["x_cat"]
        x_num = batch["x_num"]
        targets = batch["targets"]

        outputs, latents = self.forward(x_cat, x_num)

        # Unpack targets
        y_dur_log1p = targets["duration_log1p"]
        y_demand = targets["demand_share"]
        y_ride = targets["rideable_type"]

        # 1) Duration reconstruction (Gaussian NLL on log1p duration)
        loss_duration = self._duration_nll(
            outputs["duration_mu"],
            outputs["duration_logvar"],
            y_dur_log1p,
        )

        # 2) Demand regression (MSE)
        loss_demand = F.mse_loss(outputs["demand_pred"], y_demand)

        # 3) Ride type classification (CrossEntropy)
        loss_ride_type = F.cross_entropy(outputs["ride_logits"], y_ride)

        # 4) KL terms
        kl_global = kl_divergence_standard_normal(
            latents["mu_global"], latents["logvar_global"]
        )
        kl_individual = kl_divergence_standard_normal(
            latents["mu_individual"], latents["logvar_individual"]
        )

        # Weighted sum
        total_loss = (
            loss_weights["w_duration"] * loss_duration
            + loss_weights["w_demand"] * loss_demand
            + loss_weights["w_ride_type"] * loss_ride_type
            + loss_weights["beta_global"] * kl_global
            + loss_weights["beta_individual"] * kl_individual
        )

        return {
            "loss": total_loss,
            "loss_duration": loss_duration,
            "loss_demand": loss_demand,
            "loss_ride_type": loss_ride_type,
            "kl_global": kl_global,
            "kl_individual": kl_individual,
        }

    # ------------------------------------------------------------------
    # Anomaly score (for Phase 6)
    # ------------------------------------------------------------------

    def compute_anomaly_score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute a simple anomaly score based on reconstruction errors.

        Current definition:
            score = duration NLL (log1p) + λ * demand MSE
        You can refine this later.

        Returns:
            scores: FloatTensor [B]
        """
        x_cat = batch["x_cat"]
        x_num = batch["x_num"]
        targets = batch["targets"]

        outputs, _ = self.forward(x_cat, x_num)

        # Duration NLL per-sample (not mean)
        target_dur = targets["duration_log1p"].view_as(outputs["duration_mu"])
        var = torch.exp(outputs["duration_logvar"])
        nll_per_sample = 0.5 * (
            math.log(2 * math.pi)
            + outputs["duration_logvar"]
            + (target_dur - outputs["duration_mu"]).pow(2) / var
        )

        # Demand MSE per-sample
        target_dem = targets["demand_share"].view_as(outputs["demand_pred"])
        mse_per_sample = (outputs["demand_pred"] - target_dem).pow(2)

        # Simple weighted sum (λ can be tuned)
        lambda_dem = 1.0
        scores = nll_per_sample + lambda_dem * mse_per_sample

        return scores
