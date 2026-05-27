import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class FastMetrics:
    def __init__(
        self,
        vocab_embeddings: torch.Tensor,
        alpha_mse: float = 10000.0,
        reldist_ratio: float = 0.5,
    ):
        """
        vocab_embeddings: [V, H] float32 tensor of all target embeddings.
        alpha_mse: weight for MSE component in legacy distance matrix.
        """
        self.embeddings = vocab_embeddings.float()
        self.V, self.H = self.embeddings.shape
        self.emb_sq_norms = (self.embeddings ** 2).sum(dim=1)
        self.norm_embeddings = F.normalize(self.embeddings, p=2, dim=1)
        self.alpha_mse = alpha_mse
        self.reldist_ratio = reldist_ratio

        # ── Precomputed statistics ──
        self.emb_mean = self.embeddings.mean(dim=0)                 # [H]
        self.emb_norms = self.embeddings.norm(dim=1)                # [V]
        self.emb_mean_norm = self.emb_norms.mean()                  # scalar

        # Centered & normalized variants
        self.centered_embeddings = self.embeddings - self.emb_mean  # [V, H]
        self.norm_centered_embeddings = F.normalize(
            self.centered_embeddings, p=2, dim=1
        )                                                            # [V, H]

        # PCA of the embedding matrix (lazy-ish: computed once in __init__)
        self._has_pca = False
        self.pca_components = None      # [H, H]
        self.pca_singular_values = None # [H]
        self.pca_weights = None         # [H]
        #self._compute_pca()

    def _compute_pca(self):
        """Compute PCA via SVD of centered embeddings. Called once in __init__."""
        try:
            # centered_embeddings: [V, H]. SVD gives U[V,H], S[H], Vh[H,H]
            _, S, Vh = torch.linalg.svd(self.centered_embeddings, full_matrices=False)
            self.pca_components = Vh                                 # [H, H]
            self.pca_singular_values = S[: self.H]                   # [H]
            # Weights ∝ singular values, normalized so sum = H (scale-neutral)
            self.pca_weights = (
                self.pca_singular_values
                / self.pca_singular_values.sum()
                * self.H
            )
            self._has_pca = True
        except Exception as e:
            print(f"[FastMetrics] PCA computation failed: {e}")
            self._has_pca = False

    # ──────────────────────────────────────────────
    #  Legacy distance matrix (kept for backward compat)
    # ──────────────────────────────────────────────
    def compute_distance_matrix(self, preds):
        """
        preds: [B, H] float32
        Returns: [B, V] distance matrix (cos_dist + alpha * mse)
        NOTE: This is a legacy metric dominated by alpha_mse.
              For functional evaluation, use logit_mrr from compute_batch_metrics.
        """
        preds = preds.float()
        pred_sq_norms = (preds ** 2).sum(dim=1, keepdim=True)
        dot = torch.mm(preds, self.embeddings.t())
        all_mse = (pred_sq_norms + self.emb_sq_norms.unsqueeze(0) - 2 * dot) / self.H
        all_mse = torch.clamp(all_mse, min=0.0)

        norm_pred = F.normalize(preds, p=2, dim=1)
        all_cos_sim = torch.mm(norm_pred, self.norm_embeddings.t())
        all_cos_dist = 1.0 - all_cos_sim

        return all_cos_dist + self.alpha_mse * all_mse

    # ──────────────────────────────────────────────
    #  Batch metrics (extended)
    # ──────────────────────────────────────────────
    @torch.no_grad()
    def compute_batch_metrics(self, preds, target_ids):
        """
        preds: [B, H]
        target_ids: [B]
        Returns dict of scalar means (all detached).
        """
        preds = preds.float()
        targets = self.embeddings[target_ids]

        B = preds.size(0)
        batch_idx = torch.arange(B, device=preds.device)

        # ── Direct metrics ──
        mse = F.mse_loss(preds, targets, reduction='none').mean(dim=1)
        cos_dist = 1.0 - F.cosine_similarity(preds, targets, dim=-1)

        # ── Legacy composite distance metrics ──
        dist_matrix = self.compute_distance_matrix(preds)
        d_target = dist_matrix[batch_idx, target_ids].clone()

        ranks = (dist_matrix < d_target.unsqueeze(1)).sum(dim=1) + 1
        mrr = (1.0 / ranks.float()).mean()

        dist_matrix[batch_idx, target_ids] = float('inf')
        d_nearest, _ = dist_matrix.min(dim=1)
        reldist = d_target / (d_target + d_nearest + 1e-9)

        # ── Norm error ──
        pred_norms = preds.norm(dim=1)
        target_norms = self.emb_norms[target_ids]
        norm_error = (pred_norms - target_norms).abs().mean()

        # ── Centered cosine distance ──
        preds_centered = preds - self.emb_mean
        targets_centered = self.centered_embeddings[target_ids]
        centered_cos_dist = (
            1.0 - F.cosine_similarity(preds_centered, targets_centered, dim=-1)
        ).mean()

        # ── Soft-logit KL at multiple temperatures ──
        pred_dots = preds @ self.embeddings.t()          # [B, V]
        target_dots = targets @ self.embeddings.t()      # [B, V]

        kl_metrics = {}
        for tau in (0.1, 0.5, 1.0):
            kl = F.kl_div(
                F.log_softmax(pred_dots / tau, dim=-1),
                F.softmax(target_dots / tau, dim=-1),
                reduction='batchmean',
            )
            kl_metrics[f"soft_kl_{tau}"] = kl

        # ── Logit-space MRR ──
        logit_ranks = (
            (pred_dots > pred_dots[batch_idx, target_ids].unsqueeze(1)).sum(dim=1) + 1
        )
        logit_mrr = (1.0 / logit_ranks.float()).mean()

        # ── Logit-space top-k accuracy ──
        top1_acc = (logit_ranks == 1).float().mean()
        top5_acc = (logit_ranks <= 5).float().mean()
        top10_acc = (logit_ranks <= 10).float().mean()

        result = {
            # Core
            "mse": mse.mean(),
            "cosine_dist": cos_dist.mean(),
            "centered_cos_dist": centered_cos_dist,
            "norm_error": norm_error,
            # Logit-space (functional)
            "logit_mrr": logit_mrr,
            "logit_top1_acc": top1_acc,
            "logit_top5_acc": top5_acc,
            "logit_top10_acc": top10_acc,
            # KL at different temperatures
            **kl_metrics,
            # Legacy
            "mrr": mrr,
            "reldist": reldist.mean(),
        }
        return result

    # ──────────────────────────────────────────────
    #  Per-token metrics (for lifecycle analysis)
    # ──────────────────────────────────────────────
    @torch.no_grad()
    def compute_per_token_metrics(self, preds, target_ids, compute_ranks=True):
        """
        preds: [B, H]
        target_ids: [B]
        compute_ranks: if False, skip O(N*V) logit rank computation (for direct mode)
        Returns dict of per-example tensors [B] (all detached, CPU).
        Same computations as compute_batch_metrics() but without .mean() reduction.
        """
        preds = preds.float()
        target_ids_cpu = target_ids.cpu()
        targets = self.embeddings[target_ids_cpu]

        B = preds.size(0)

        # ── Direct metrics ──
        mse = F.mse_loss(preds, targets.to(preds.device), reduction='none').mean(dim=1)  # [B]
        cos_dist = 1.0 - F.cosine_similarity(preds, targets.to(preds.device), dim=-1)  # [B]

        # ── Centered cosine distance ──
        preds_centered = preds - self.emb_mean.to(preds.device)
        targets_centered = self.centered_embeddings[target_ids_cpu].to(preds.device)
        centered_cos_dist = 1.0 - F.cosine_similarity(preds_centered, targets_centered, dim=-1)  # [B]

        # ── Norm error ──
        pred_norms = preds.norm(dim=1)  # [B]
        target_norms = self.emb_norms[target_ids_cpu]  # [B]
        norm_error = (pred_norms.cpu() - target_norms).abs() / (target_norms + 1e-9)  # [B], relative

        # ── Logit-space ranks (only for head mode on base tokens) ──
        logit_ranks = None
        if compute_ranks:
            pred_dots = preds @ self.embeddings.to(preds.device).t()  # [B, V]
            batch_idx = torch.arange(B, device=pred_dots.device)
            logit_ranks = (
                (pred_dots > pred_dots[batch_idx, target_ids_cpu.to(pred_dots.device)].unsqueeze(1)).sum(dim=1) + 1
            )  # [B]

        result = {
            "mse": mse.cpu(),
            "cos_dist": cos_dist.cpu(),
            "centered_cos_dist": centered_cos_dist.cpu(),
            "norm_error": norm_error.cpu(),
            "pred_norms": pred_norms.cpu(),
            "target_norms": target_norms.cpu(),
        }
        if logit_ranks is not None:
            result["logit_ranks"] = logit_ranks.cpu()
        return result

    # ──────────────────────────────────────────────
    #  Loss building blocks (private helpers)
    # ──────────────────────────────────────────────
    def _cosine_loss(self, preds, targets):
        """Standard cosine distance loss."""
        return (1.0 - F.cosine_similarity(preds, targets, dim=-1)).mean()

    def _centered_cosine_loss(self, preds, target_ids):
        """Cosine loss on mean-centered embeddings. Removes anisotropy bias."""
        preds_c = preds - self.emb_mean
        targets_c = self.centered_embeddings[target_ids]
        return (1.0 - F.cosine_similarity(preds_c, targets_c, dim=-1)).mean()

    def _norm_loss(self, preds, target_ids):
        """MSE between L2 norms of predicted and target embeddings."""
        pred_norms = preds.norm(dim=1)
        target_norms = self.emb_norms[target_ids]
        return F.mse_loss(pred_norms, target_norms)

    def _soft_logit_kl_loss(self, preds, target_ids, tau=0.1):
        """
        KL(softmax(E·e*/τ) || softmax(E·ê/τ)).

        Optimizes functional equivalence: the softmax distribution
        induced by the embedding in the LM head should match gold.

        Target distribution is detached (no gradients through gold embeddings).
        """
        pred_logits = preds @ self.embeddings.t() / tau        # [B, V]
        with torch.no_grad():
            targets = self.embeddings[target_ids]
            target_logits = targets @ self.embeddings.t() / tau  # [B, V]
            target_probs = F.softmax(target_logits, dim=-1)

        return F.kl_div(
            F.log_softmax(pred_logits, dim=-1),
            target_probs,
            reduction='batchmean',
        )

    def _multi_tau_kl_loss(self, preds, target_ids, taus=(0.05, 0.1, 0.5, 2.0)):
        """
        Multi-scale soft-logit KL.
        Small τ → local neighborhood fidelity. Large τ → global positioning.
        Dot products computed once, softmax applied per-τ.
        """
        pred_dots = preds @ self.embeddings.t()  # [B, V]
        with torch.no_grad():
            targets = self.embeddings[target_ids]
            target_dots = targets @ self.embeddings.t()  # [B, V]

        total = 0.0
        for tau in taus:
            with torch.no_grad():
                target_probs = F.softmax(target_dots / tau, dim=-1)
            kl = F.kl_div(
                F.log_softmax(pred_dots / tau, dim=-1),
                target_probs,
                reduction='batchmean',
            )
            total = total + kl
        return total / len(taus)

    def _pca_weighted_mse_loss(self, preds, target_ids):
        """
        MSE weighted by PCA singular values.
        First principal components (high variance) get higher weight.
        Falls back to plain MSE if PCA unavailable.
        """
        if not self._has_pca:
            return F.mse_loss(preds, self.embeddings[target_ids])

        targets = self.embeddings[target_ids]
        diff = preds - targets                          # [B, H]
        proj = diff @ self.pca_components.t()           # [B, H]
        weighted_sq = proj ** 2 * self.pca_weights.unsqueeze(0)
        return weighted_sq.mean()

    def _contrastive_logit_loss(self, preds, target_ids, tau=0.1, n_negatives=1024,
                                 norm_reg=0.0):
        """
        Sampled InfoNCE in logit space with hard-negative mining.
        Optional norm regularization to prevent norm explosion.

        Args:
            norm_reg: weight for norm matching regularizer (0 = disabled).
        """
        B = preds.size(0)
        batch_idx = torch.arange(B, device=preds.device)

        all_dots = preds @ self.embeddings.t()  # [B, V]
        pos_logits = all_dots[batch_idx, target_ids]   # [B]

        # Mask positives for hard-negative mining (in-place + restore)
        orig_vals = pos_logits.clone()
        all_dots[batch_idx, target_ids] = -float('inf')

        k = min(n_negatives, self.V - 1)
        neg_logits, _ = all_dots.topk(k, dim=1)        # [B, k]
        all_dots[batch_idx, target_ids] = orig_vals     # restore

        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1) / tau
        labels = torch.zeros(B, device=preds.device, dtype=torch.long)
        ce = F.cross_entropy(logits, labels)

        if norm_reg > 0:
            ce = ce + norm_reg * self._norm_loss(preds, target_ids)

        return ce

    # ──────────────────────────────────────────────
    #  Loss dispatcher
    # ──────────────────────────────────────────────
    def compute_loss(self, preds, target_ids, loss_type="mse", **kwargs):
        """
        Returns scalar loss tensor for training.

        Active loss types (recommended):
          Tier 1 — Baselines:
            "mse"               : Plain MSE
            "cosine"            : Cosine distance
            "centered_cosine"   : Cosine on centered embeddings
            "cosine_norm"       : Cosine + norm matching (kwargs: alpha_norm)
            "pca_weighted_mse"  : PCA-weighted MSE

          Tier 2 — Functional (logit-space):
            "soft_logit_kl"     : KL on softmax logit distributions (kwargs: tau)
            "multi_tau_kl"      : Multi-scale KL (kwargs: taus)
            "contrastive_logit" : Hard-neg InfoNCE (kwargs: tau, n_negatives, norm_reg)

          Tier 3 — Composite:
            "geometric_aware"   : centered_cos + norm + soft_logit_kl (kwargs: alpha, beta, gamma, tau)
            "full_spectrum"     : multi_tau_kl + centered_cos + norm + pca_mse (kwargs: w_kl, w_cos, w_norm, w_pca, taus)

          Legacy (kept for backward compatibility):
            "mse_cosine", "reldist", "reldist_mse_cosine", "reldist_mse_cosine_mult",
            "triplet_composite", "infonce_*" variants
        """
        preds = preds.float()
        targets = self.embeddings[target_ids]

        # ══════════════════════════════════════════
        #  TIER 1 — Baselines
        # ══════════════════════════════════════════
        if loss_type == "mse":
            return F.mse_loss(preds, targets)

        elif loss_type == "cosine":
            return self._cosine_loss(preds, targets)

        elif loss_type == "centered_cosine":
            return self._centered_cosine_loss(preds, target_ids)

        elif loss_type == "cosine_norm":
            # Centered cosine + norm matching (fix: use centered, not plain)
            alpha_norm = kwargs.get("alpha_norm", 1.0)
            return (
                self._centered_cosine_loss(preds, target_ids)
                + alpha_norm * self._norm_loss(preds, target_ids)
            )

        elif loss_type == "pca_weighted_mse":
            return self._pca_weighted_mse_loss(preds, target_ids)

        # ══════════════════════════════════════════
        #  TIER 2 — Functional (logit-space)
        # ══════════════════════════════════════════
        elif loss_type == "soft_logit_kl":
            tau = kwargs.get("tau", 0.1)
            return self._soft_logit_kl_loss(preds, target_ids, tau=tau)

        elif loss_type == "multi_tau_kl":
            taus = kwargs.get("taus", (0.05, 0.1, 0.5, 2.0))
            return self._multi_tau_kl_loss(preds, target_ids, taus=taus)

        elif loss_type == "contrastive_logit":
            tau = kwargs.get("tau", 0.1)
            n_neg = kwargs.get("n_negatives", 1024)
            norm_reg = kwargs.get("norm_reg", 1.0)  # default: norm protection ON
            return self._contrastive_logit_loss(
                preds, target_ids, tau=tau, n_negatives=n_neg, norm_reg=norm_reg
            )

        # ══════════════════════════════════════════
        #  TIER 3 — Composite
        # ══════════════════════════════════════════
        elif loss_type == "geometric_aware":
            alpha = kwargs.get("alpha", 1.0)
            beta = kwargs.get("beta", 1.0)
            gamma = kwargs.get("gamma", 1.0)
            tau = kwargs.get("tau", 0.1)

            l_cos = self._centered_cosine_loss(preds, target_ids)
            l_norm = self._norm_loss(preds, target_ids)
            l_kl = self._soft_logit_kl_loss(preds, target_ids, tau=tau)

            return alpha * l_cos + beta * l_norm + gamma * l_kl

        elif loss_type == "full_spectrum":
            w_kl = kwargs.get("w_kl", 2.0)
            w_cos = kwargs.get("w_cos", 1.0)
            w_norm = kwargs.get("w_norm", 0.3)
            w_pca = kwargs.get("w_pca", 0.5)
            taus = kwargs.get("taus", (0.05, 0.1, 0.5, 2.0))

            l_kl = self._multi_tau_kl_loss(preds, target_ids, taus=taus)
            l_cos = self._centered_cosine_loss(preds, target_ids)
            l_norm = self._norm_loss(preds, target_ids)
            l_pca = self._pca_weighted_mse_loss(preds, target_ids)

            return w_kl * l_kl + w_cos * l_cos + w_norm * l_norm + w_pca * l_pca
        elif loss_type == "mse_plus_cosine":
            """MSE-dominated with a touch of centered cosine for direction."""
            alpha_cos = kwargs.get("alpha_cos", 0.01)
            l_mse = F.mse_loss(preds, targets)
            l_cos = self._centered_cosine_loss(preds, target_ids)
            return l_mse + alpha_cos * l_cos
        elif loss_type == "mse_norm":
            """MSE with explicit norm matching."""
            beta_norm = kwargs.get("beta_norm", 1.0)
            l_mse = F.mse_loss(preds, targets)
            l_norm = self._norm_loss(preds, target_ids)
            return l_mse + beta_norm * l_norm
        # ══════════════════════════════════════════
        #  LEGACY (backward compatibility, not recommended)
        # ══════════════════════════════════════════
        elif loss_type == "mse_cosine":
            return self._cosine_loss(preds, targets) + self.alpha_mse * F.mse_loss(preds, targets)

        elif loss_type in (
            "reldist", "reldist_mse_cosine",
            "reldist_mse_cosine_mult", "triplet_composite",
        ):
            dist_matrix = self.compute_distance_matrix(preds)
            B = preds.size(0)
            batch_idx = torch.arange(B, device=preds.device)
            d_target = dist_matrix[batch_idx, target_ids].clone()

            if loss_type == "reldist":
                dist_matrix[batch_idx, target_ids] = float('inf')
                d_nearest, _ = dist_matrix.min(dim=1)
                return (d_target / (d_target + d_nearest + 1e-9)).mean()

            elif loss_type == "reldist_mse_cosine":
                dist_matrix[batch_idx, target_ids] = float('inf')
                d_nearest, _ = dist_matrix.min(dim=1)
                rel_dist_loss = (d_target / (d_target + d_nearest + 1e-9)).mean()
                mse_loss = F.mse_loss(preds, targets)
                cos_loss = self._cosine_loss(preds, targets)
                return (
                    self.reldist_ratio * rel_dist_loss
                    + (1 - self.reldist_ratio) * (cos_loss + self.alpha_mse * mse_loss)
                )

            elif loss_type == "reldist_mse_cosine_mult":
                dist_matrix[batch_idx, target_ids] = float('inf')
                d_nearest, _ = dist_matrix.min(dim=1)
                rel_dist_loss = (d_target / (d_target + d_nearest + 1e-9)).mean()
                mse_loss = F.mse_loss(preds, targets)
                cos_loss = self._cosine_loss(preds, targets)
                return rel_dist_loss * (cos_loss + self.alpha_mse * mse_loss)

            elif loss_type == "triplet_composite":
                margin = kwargs.get("margin", 0.1)
                dist_matrix[batch_idx, target_ids] = float('inf')
                d_nearest, _ = dist_matrix.min(dim=1)
                triplet_loss = F.relu(d_target - d_nearest + margin).mean()
                return triplet_loss + d_target.mean()

        elif loss_type.startswith("infonce"):
            tau = kwargs.get("tau", 0.05)

            if loss_type in ("infonce_cosine", "infonce_cosine_anchored", "infonce_margin_anchored"):
                norm_pred = F.normalize(preds, p=2, dim=1)
                cos_sim = torch.mm(norm_pred, self.norm_embeddings.t())
                if loss_type == "infonce_margin_anchored":
                    margin = kwargs.get("margin", 0.1)
                    batch_idx = torch.arange(preds.size(0), device=preds.device)
                    cos_sim[batch_idx, target_ids] -= margin
                logits = cos_sim / tau

            elif loss_type in ("infonce_composite_anchored", "infonce_composite_margin_anchored"):
                dist_matrix = self.compute_distance_matrix(preds)
                if loss_type == "infonce_composite_margin_anchored":
                    margin = kwargs.get("margin", 0.1)
                    batch_idx = torch.arange(preds.size(0), device=preds.device)
                    dist_matrix[batch_idx, target_ids] += margin
                logits = -dist_matrix / tau

            elif loss_type == "infonce_mse":
                pred_sq_norms = (preds ** 2).sum(dim=1, keepdim=True)
                dot = torch.mm(preds, self.embeddings.t())
                all_mse = (pred_sq_norms + self.emb_sq_norms.unsqueeze(0) - 2 * dot) / self.H
                all_mse = torch.clamp(all_mse, min=0.0)
                logits = -all_mse / tau
            else:
                raise ValueError(f"Unknown infonce loss_type: {loss_type}")

            ce_loss = F.cross_entropy(logits, target_ids)

            is_anchored = loss_type in (
                "infonce_cosine_anchored",
                "infonce_margin_anchored",
                "infonce_composite_anchored",
                "infonce_composite_margin_anchored",
            )
            if is_anchored:
                mse_loss = F.mse_loss(preds, targets)
                return ce_loss + self.alpha_mse * mse_loss

            return ce_loss

        raise ValueError(f"Unknown loss_type: {loss_type}")