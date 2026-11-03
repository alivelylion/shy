import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ----------------------------
# Minimal GroupPartition class
# ----------------------------
class GroupPartition:
    """
    Splits the last (channel) dimension into 'partitions' groups.
    The partition() method reshapes an input tensor of shape
    (batch, height, width, channels) into a flattened tensor
    where each row corresponds to a grouped portion of the channels.
    The unpartition() method reverses this operation.
    """

    def __init__(self, partitions):
        self.partitions = partitions

    def partition(self, x):
        # x: (b, h, w, c), with c divisible by partitions.
        b, h, w, c = x.shape
        assert c % self.partitions == 0, "Channel dim must be divisible by partitions"
        new_c = c // self.partitions
        # reshape to (b, h, w, partitions, new_c) then bring partitions forward
        x = x.view(b, h, w, self.partitions, new_c)
        x = x.permute(0, 3, 1, 2, 4).contiguous()  # (b, partitions, h, w, new_c)
        # Flatten out the batch, partitions, h and w into one dimension
        x = x.view(-1, new_c)  # (b * partitions * h * w, new_c)
        return x

    def unpartition(self, x, original_shape):
        # x: (b * partitions * h * w, new_c), original_shape: (b, h, w, c)
        b, h, w, c = original_shape
        new_c = c // self.partitions
        # Reshape back to (b, partitions, h, w, new_c)
        x = x.view(b, self.partitions, h, w, new_c)
        # Permute back to (b, h, w, partitions, new_c) and then flatten partitions into channels
        x = x.permute(0, 2, 3, 1, 4).contiguous()  # (b, h, w, partitions, new_c)
        x = x.view(b, h, w, c)  # (b, h, w, c)
        return x


# ----------------------------
# KeplerLoss module
# ----------------------------
class KeplerLoss(nn.Module):
    """
    Implements an auxiliary loss as described in the Kepler Codebook paper.
    When enabled (use=True), it computes a KL divergence loss between
    the softmax-normalized quantized latents and a pre-computed high-dimensional grid.
    """

    def __init__(self, use, kl_weight, n_e):
        super(KeplerLoss, self).__init__()
        self.use = use
        self.kl_weight = kl_weight
        self.n_e = n_e
        self.prior_prob = self.create_high_dimensional_grid()

    def create_high_dimensional_grid(self):
        num_points = 2048
        dimensions = 64
        sub_dimensions = 16

        points_per_dim = int(np.ceil(num_points ** (1 / sub_dimensions)))
        low_dim_grid = (
            np.indices([points_per_dim] * sub_dimensions).reshape(sub_dimensions, -1).T
        )
        high_dim_grid = np.tile(low_dim_grid, (1, dimensions // sub_dimensions))

        if high_dim_grid.shape[0] < num_points:
            raise ValueError(
                "Insufficient points to generate the required sphere centers."
            )

        start_index = np.random.randint(0, high_dim_grid.shape[0] - self.n_e + 1)
        selected_points = high_dim_grid[start_index : start_index + self.n_e, :]

        return torch.from_numpy(selected_points).float()

    def forward(self, z):
        # Flatten z: (batch, -1)
        p = z.view(z.shape[0], -1)
        p = (p - p.mean(dim=1, keepdim=True)) / p.var(dim=1, keepdim=True)
        p = F.softmax(p, dim=1)

        q = self.prior_prob.reshape(1, -1).repeat(p.shape[0], 1)
        q = (q - q.mean(dim=1, keepdim=True)) / q.var(dim=1, keepdim=True)
        q = F.softmax(q[:, : p.shape[1]], dim=1).to(p.device)
        # Compute the KL divergence loss
        kl_loss = F.kl_div(p.log(), q, reduction="batchmean") * self.kl_weight
        return kl_loss


# ----------------------------
# KeplerQuantizer module
# ----------------------------
class KeplerQuantizer(nn.Module):
    """
    A minimal vector quantizer that implements the Kepler Codebook approach.

    It partitions the latent representation, finds the nearest embedding vectors,
    and computes the VQ loss with a stop-gradient trick. Optionally, an additional
    Kepler loss (e.g. KL divergence) is applied.

    Args:
        embed_dim: Dimension of the input latent vector.
        scale: Scaling factor for the number of embeddings (k in paper).
        partitions: Number of groups to partition the latent (d in paper).
        n_embed: Base number of embeddings (K in paper).
        beta: Commitment loss coefficient.
        kepler_loss: An instance of KeplerLoss.
        legacy: If True, use the “legacy” loss weighting.
    """

    def __init__(
        self,
        embed_dim,
        scale,
        partitions,
        n_embed,
        beta,
        kepler_loss: KeplerLoss,
        legacy=True,
    ):
        super(KeplerQuantizer, self).__init__()
        self.scale = scale
        self.partitions = partitions
        self.n_embed = n_embed  # base embeddings count (K)
        self.n_e = int(n_embed * scale)  # scaled number of embeddings
        self.e_dim = embed_dim // partitions  # dimension per partition
        self.beta = beta
        self.legacy = legacy
        self.kepler_loss = kepler_loss

        # Create embedding table with shape (n_e, e_dim)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # Create a partition utility instance
        self.partition = GroupPartition(partitions)

    def forward(self, z):
        """
        Forward pass for vector quantization.

        z: Input tensor from the VAE encoder of shape (batch, embed_dim, height, width).

        Returns:
            z_q: Quantized tensor, same shape as input.
            loss: Quantization loss (including optional Kepler loss).
            info: Additional information (here left as None).
        """
        # Rearrange input to (b, h, w, embed_dim)
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        original_shape = z_perm.shape  # (b, h, w, embed_dim)

        # Partition the latent representations: result shape (N, e_dim)
        z_flattened = self.partition.partition(z_perm)

        # Compute squared Euclidean distances between z vectors and embeddings:
        #   d = ||z||^2 + ||e||^2 - 2 * z @ e.T
        z_sq = torch.sum(z_flattened**2, dim=1, keepdim=True)  # (N, 1)
        e_sq = torch.sum(self.embedding.weight**2, dim=1)  # (n_e)
        distances = (
            z_sq + e_sq - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Get the index of the nearest embedding for each vector
        encoding_indices = torch.argmin(distances, dim=1)  # (N)

        # Retrieve quantized vectors from the embedding table
        z_q_flattened = self.embedding(encoding_indices)  # (N, e_dim)

        # Unpartition to reconstruct the quantized output with shape (b, h, w, embed_dim)
        z_q = self.partition.unpartition(z_q_flattened, original_shape)

        # Compute the VQ loss using the stop-gradient trick
        if self.legacy:
            loss = torch.mean((z_q.detach() - z_perm) ** 2) + self.beta * torch.mean(
                (z_q - z_perm.detach()) ** 2
            )
        else:
            loss = self.beta * torch.mean((z_q.detach() - z_perm) ** 2) + torch.mean(
                (z_q - z_perm.detach()) ** 2
            )

        # Optionally add the additional Kepler loss if enabled
        if self.kepler_loss.use:
            loss += self.kepler_loss(z_q)

        # Preserve gradients by adding back the difference (the so-called "straight-through" estimator)
        z_q = z_perm + (z_q - z_perm).detach()

        # Rearrange back to the original input shape: (b, embed_dim, h, w)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss


# ----------------------------
# Minimal usage example
# ----------------------------
if __name__ == "__main__":
    # Example configuration (as provided by the official team):
    embed_dim = 256
    scale = 1  # k = n_embed when scale == 1
    partitions = 4  # partition number d in the paper
    n_embed = 1024  # K in the paper
    beta = 0.25  # Example commitment loss weight

    # Kepler loss configuration: disable for now (set use=False) or enable as needed.
    kepler_loss = KeplerLoss(use=False, kl_weight=1e-8, n_e=n_embed)

    # Instantiate the KeplerQuantizer with the configuration above.
    quantizer = KeplerQuantizer(
        embed_dim=embed_dim,
        scale=scale,
        partitions=partitions,
        n_embed=n_embed,
        beta=beta,
        kepler_loss=kepler_loss,
        legacy=True,
    )

    # Create a dummy latent tensor from the VAE encoder
    dummy_input = torch.randn(8, embed_dim, 16, 16)  # batch=8, spatial resolution 16x16

    # Forward pass: get quantized output and loss.
    z_q, loss, info = quantizer(dummy_input)

    print("Quantized output shape:", z_q.shape)  # Expected: (8, 256, 16, 16)
    print("Quantization loss:", loss.item())