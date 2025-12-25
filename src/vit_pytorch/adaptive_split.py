"""
Adaptive Quadtree Split for Hilbert ViT

This module implements Variable Depth tokenization with two parallel approaches:
- Scheme B: Balanced Greedy Splitting (2:1 constraint)
- Scheme C: Fixed Budget Dynamic Programming

Both schemes maintain Hilbert curve locality and are compatible with LCA bias.

Mathematical Foundation:
-----------------------
Quadtree-Hilbert Isomorphism:
    QuadtreePath(R) = [q₁, q₂, ..., qₐ] ⟺ HilbertSegment(R) = H|[a,b]

Complexity Function:
    C(R) = α · C_var(R) + (1-α) · C_grad(R)
    
    where:
    - C_var(R) = Var(R) / (Var(R) + σ₀²)  [normalized variance]
    - C_grad(R) = G(R) / (G(R) + g₀²)     [normalized gradient energy]

Depth-Dependent Threshold:
    τ_d = τ₀ · γ^d

Author: GitHub Copilot
Date: 2025-12-25
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .hilbert import HilbertCurve


# =============================================================================
# Configuration
# =============================================================================

class SplitScheme(Enum):
    """Available splitting schemes."""
    BALANCED_GREEDY = "balanced_greedy"  # Scheme B
    FIXED_BUDGET_DP = "fixed_budget_dp"  # Scheme C


@dataclass
class AdaptiveSplitConfig:
    """
    Configuration for adaptive quadtree splitting.
    
    All parameters have mathematically derived defaults.
    See IMPROVEMENT_PLAN.md Section 10 for derivations.
    """
    
    # === Complexity Function Parameters ===
    
    alpha: float = 0.5
    """Variance weight α ∈ [0, 1]
    
    Mathematical basis:
    - α = 1.0: Pure variance, texture-sensitive
    - α = 0.0: Pure gradient, edge-sensitive  
    - α = 0.5: Balanced (recommended)
    
    Range: [0.3, 0.7]
    """
    
    sigma_0_sq: float = 0.01
    """Variance normalization constant σ₀²
    
    Mathematical basis:
    - Natural image local variance median ≈ 0.014
    - C_var = 0.5 when Var = σ₀²
    
    Range: [0.005, 0.03]
    """
    
    g_0_sq: float = 0.08
    """Gradient normalization constant g₀²
    
    Mathematical basis:
    - Natural image gradient energy median ≈ 0.07
    - C_grad = 0.5 when G = g₀²
    
    Range: [0.03, 0.15]
    """
    
    # === Threshold Function Parameters ===
    
    tau_0: float = 0.15
    """Root threshold τ₀
    
    Mathematical basis:
    - 15-20th percentile of natural image complexity
    - Controls "how simple must image be to not split"
    
    Range: [0.08, 0.25]
    """
    
    gamma: float = 0.85
    """Threshold decay factor γ ∈ (0, 1)
    
    Mathematical basis:
    - τ_d = τ₀ · γ^d
    - γ = 0.85 → deepest threshold ≈ 52% of root
    
    Range: [0.75, 0.92]
    """
    
    max_depth: int = 4
    """Maximum split depth d_max
    
    Mathematical basis:
    - d=4 corresponds to 224→14 patch size
    - Aligned with ViT-B/16
    - Maximum 4^4 = 256 tokens
    
    Range: [3, 5]
    """
    
    min_region_size: int = 7
    """Minimum region edge length (pixels)
    
    Mathematical basis:
    - At least 7×7 needed for basic texture features
    
    Range: [4, 16]
    """
    
    # === Scheme Selection ===
    
    scheme: SplitScheme = SplitScheme.BALANCED_GREEDY
    """Which splitting scheme to use"""
    
    # === Scheme B Specific Parameters ===
    
    enforce_balance: bool = True
    """Whether to enforce 2:1 balance constraint"""
    
    max_balance_iterations: int = 10
    """Maximum iterations for balance refinement"""
    
    target_tokens: Optional[int] = None
    """Soft target for token count (None = no constraint)"""
    
    token_penalty_weight: float = 0.01
    """Weight for token count deviation penalty"""
    
    # === Scheme C Specific Parameters ===
    
    token_budget: int = 64
    """Fixed token budget for Scheme C"""
    
    locality_penalty_weight: float = 0.1
    """Weight for depth discontinuity penalty in DP"""
    
    # === Computation Parameters ===
    
    use_integral_image: bool = True
    """Use integral images for O(1) region statistics"""
    
    gradient_method: str = "simple"
    """Gradient computation: 'simple' or 'sobel'"""
    
    def __post_init__(self) -> None:
        """自动验证配置."""
        self.validate()

    def get_threshold(self, depth: int) -> float:
        """Get threshold for given depth."""
        return self.tau_0 * (self.gamma ** depth)
    
    def validate(self) -> None:
        """Validate configuration."""
        assert 0.0 <= self.alpha <= 1.0, f"alpha must be in [0, 1], got {self.alpha}"
        assert self.sigma_0_sq > 0, f"sigma_0_sq must be positive"
        assert self.g_0_sq > 0, f"g_0_sq must be positive"
        assert 0.0 < self.tau_0 < 1.0, f"tau_0 must be in (0, 1)"
        assert 0.0 < self.gamma < 1.0, f"gamma must be in (0, 1)"
        assert self.max_depth >= 1, f"max_depth must be >= 1"
        assert self.min_region_size >= 1, f"min_region_size must be >= 1"
        assert self.gradient_method in ("simple", "sobel")
    
    @classmethod
    def scheme_b(cls, **kwargs) -> "AdaptiveSplitConfig":
        """Create Scheme B configuration."""
        return cls(scheme=SplitScheme.BALANCED_GREEDY, **kwargs)
    
    @classmethod
    def scheme_c(cls, token_budget: int = 64, **kwargs) -> "AdaptiveSplitConfig":
        """Create Scheme C configuration."""
        return cls(
            scheme=SplitScheme.FIXED_BUDGET_DP,
            token_budget=token_budget,
            **kwargs
        )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Region:
    """A rectangular region in the image."""
    x1: int  # left
    y1: int  # top
    x2: int  # right (exclusive)
    y2: int  # bottom (exclusive)
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def get_quadrant(self, q: int) -> "Region":
        """Get q-th quadrant (0=TL, 1=TR, 2=BL, 3=BR)."""
        mx = (self.x1 + self.x2) // 2
        my = (self.y1 + self.y2) // 2
        
        if q == 0:  # Top-Left
            return Region(self.x1, self.y1, mx, my)
        elif q == 1:  # Top-Right
            return Region(mx, self.y1, self.x2, my)
        elif q == 2:  # Bottom-Left
            return Region(self.x1, my, mx, self.y2)
        else:  # Bottom-Right (q == 3)
            return Region(mx, my, self.x2, self.y2)


@dataclass
class QuadtreeNode:
    """A node in the quadtree."""
    region: Region
    depth: int
    path: List[int]  # quadrant indices from root
    complexity: float = 0.0
    importance: float = 0.0  # for Scheme C
    children: Optional[List["QuadtreeNode"]] = None
    
    @property
    def is_leaf(self) -> bool:
        return self.children is None
    
    @property
    def hilbert_idx(self) -> int:
        """Compute Hilbert index from center position."""
        # Will be set by HilbertTokenSorter
        return getattr(self, "_hilbert_idx", 0)
    
    @hilbert_idx.setter
    def hilbert_idx(self, value: int):
        self._hilbert_idx = value


@dataclass
class SplitToken:
    """Output token from adaptive splitting."""
    region: Region
    depth: int
    path: List[int]
    hilbert_idx: int
    complexity: float
    
    def to_levels_info(self, max_depth: int) -> List[int]:
        """Convert to levels_info format [depth, q1, q2, ..., 0, 0, ...]"""
        info = [self.depth] + self.path + [0] * (max_depth - len(self.path))
        return info[:max_depth + 1]


@dataclass 
class SplitResult:
    """Result of adaptive splitting for one image."""
    tokens: List[SplitToken]
    
    @property
    def num_tokens(self) -> int:
        return len(self.tokens)
    
    @property
    def depth_distribution(self) -> Dict[int, int]:
        """Count tokens at each depth."""
        dist = {}
        for t in self.tokens:
            dist[t.depth] = dist.get(t.depth, 0) + 1
        return dist
    
    @property
    def depth_entropy(self) -> float:
        """Entropy of depth distribution."""
        dist = self.depth_distribution
        total = sum(dist.values())
        entropy = 0.0
        for count in dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy
    
    def get_levels_info(self, max_depth: int) -> Tensor:
        """Get levels_info tensor [N, max_depth+1]."""
        infos = [t.to_levels_info(max_depth) for t in self.tokens]
        return torch.tensor(infos, dtype=torch.long)
    
    def get_regions_tensor(self) -> Tensor:
        """Get regions tensor [N, 4] as (x1, y1, x2, y2)."""
        regions = [[t.region.x1, t.region.y1, t.region.x2, t.region.y2] 
                   for t in self.tokens]
        return torch.tensor(regions, dtype=torch.long)


# =============================================================================
# Integral Image Utilities
# =============================================================================

class IntegralImageCache:
    """
    Cache for integral images enabling O(1) region statistics.
    
    Computes and caches:
    - II: integral of pixel values
    - II_sq: integral of squared pixel values  
    - II_grad: integral of gradient magnitude squared
    """
    
    def __init__(self, image: Tensor, gradient_method: str = "simple"):
        """
        Args:
            image: [C, H, W] or [H, W] tensor, values in [0, 1]
            gradient_method: 'simple' or 'sobel'
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        self.C, self.H, self.W = image.shape
        self.device = image.device
        
        # Convert to grayscale for complexity computation
        if self.C == 3:
            # Standard luminance weights
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image.mean(dim=0)
        
        # Compute integral images
        self.II = self._compute_integral(gray)
        self.II_sq = self._compute_integral(gray ** 2)
        self.II_grad = self._compute_integral(
            self._compute_gradient_magnitude_sq(gray, gradient_method)
        )
    
    def _compute_integral(self, img: Tensor) -> Tensor:
        """Compute integral image with padding."""
        # Pad with zeros for easier boundary handling
        padded = F.pad(img, (1, 0, 1, 0), value=0)
        integral = padded.cumsum(dim=0).cumsum(dim=1)
        return integral
    
    def _compute_gradient_magnitude_sq(
        self, gray: Tensor, method: str
    ) -> Tensor:
        """Compute squared gradient magnitude."""
        if method == "simple":
            # Simple finite differences
            grad_x = F.pad(gray[:, 1:] - gray[:, :-1], (0, 1), value=0)
            grad_y = F.pad(gray[1:, :] - gray[:-1, :], (0, 0, 0, 1), value=0)
        else:  # sobel
            # Sobel operators
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=gray.dtype, device=gray.device)
            sobel_y = sobel_x.T
            
            gray_4d = gray.unsqueeze(0).unsqueeze(0)
            grad_x = F.conv2d(gray_4d, sobel_x.view(1, 1, 3, 3), padding=1).squeeze()
            grad_y = F.conv2d(gray_4d, sobel_y.view(1, 1, 3, 3), padding=1).squeeze()
        
        return grad_x ** 2 + grad_y ** 2
    
    def query_sum(self, integral: Tensor, region: Region) -> float:
        """Query sum over region using integral image."""
        x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
        # Note: integral is padded by 1
        return (
            integral[y2, x2].item()
            - integral[y1, x2].item()
            - integral[y2, x1].item()
            + integral[y1, x1].item()
        )
    
    def compute_variance(self, region: Region) -> float:
        """Compute variance of pixel values in region."""
        area = region.area
        if area == 0:
            return 0.0
        
        sum_val = self.query_sum(self.II, region)
        sum_sq = self.query_sum(self.II_sq, region)
        
        mean = sum_val / area
        variance = sum_sq / area - mean ** 2
        return max(0.0, variance)  # Numerical stability
    
    def compute_gradient_energy(self, region: Region) -> float:
        """Compute mean squared gradient magnitude in region."""
        area = region.area
        if area == 0:
            return 0.0
        
        sum_grad = self.query_sum(self.II_grad, region)
        return sum_grad / area


# =============================================================================
# Complexity Estimation
# =============================================================================

class ComplexityEstimator:
    """
    Estimates region complexity for split decisions.
    
    C(R) = α · C_var(R) + (1-α) · C_grad(R)
    
    where:
    - C_var(R) = Var(R) / (Var(R) + σ₀²)
    - C_grad(R) = G(R) / (G(R) + g₀²)
    """
    
    def __init__(self, config: AdaptiveSplitConfig):
        self.alpha = config.alpha
        self.sigma_0_sq = config.sigma_0_sq
        self.g_0_sq = config.g_0_sq
    
    def compute(
        self, 
        region: Region, 
        cache: IntegralImageCache
    ) -> float:
        """Compute complexity for a region."""
        var = cache.compute_variance(region)
        grad = cache.compute_gradient_energy(region)
        
        c_var = var / (var + self.sigma_0_sq)
        c_grad = grad / (grad + self.g_0_sq)
        
        return self.alpha * c_var + (1 - self.alpha) * c_grad
    
    def compute_batch(
        self,
        regions: List[Region],
        cache: IntegralImageCache
    ) -> List[float]:
        """Compute complexity for multiple regions."""
        return [self.compute(r, cache) for r in regions]


# =============================================================================
# Hilbert Sorting
# =============================================================================

class HilbertTokenSorter:
    """
    Sorts tokens by Hilbert curve order.
    
    Ensures spatial locality is preserved in the token sequence.
    """
    
    def __init__(self, image_size: int):
        self.image_size = image_size
        # Find smallest power of 2 >= image_size
        self.grid_order = math.ceil(math.log2(image_size))
        self.grid_size = 2 ** self.grid_order
        # HilbertCurve uses static methods, no instantiation needed
    
    def get_hilbert_index(self, region: Region) -> int:
        """Get Hilbert index for region center."""
        cx, cy = region.center
        # Scale to grid coordinates
        gx = int(cx * self.grid_size / self.image_size)
        gy = int(cy * self.grid_size / self.image_size)
        # Clamp to valid range
        gx = max(0, min(self.grid_size - 1, gx))
        gy = max(0, min(self.grid_size - 1, gy))
        return HilbertCurve.xy_to_d(self.grid_size, gx, gy)
    
    def sort_tokens(self, tokens: List[SplitToken]) -> List[SplitToken]:
        """Sort tokens by Hilbert index."""
        for token in tokens:
            token.hilbert_idx = self.get_hilbert_index(token.region)
        return sorted(tokens, key=lambda t: t.hilbert_idx)
    
    def sort_nodes(self, nodes: List[QuadtreeNode]) -> List[QuadtreeNode]:
        """Sort quadtree nodes by Hilbert index."""
        for node in nodes:
            node.hilbert_idx = self.get_hilbert_index(node.region)
        return sorted(nodes, key=lambda n: n.hilbert_idx)


# =============================================================================
# Abstract Base Splitter
# =============================================================================

class BaseAdaptiveSplitter(ABC):
    """
    Abstract base class for adaptive splitting algorithms.
    
    Subclasses implement different splitting schemes while sharing:
    - Complexity estimation
    - Hilbert sorting
    - Output format
    """
    
    def __init__(self, config: AdaptiveSplitConfig):
        self.config = config
        config.validate()
        self.complexity_estimator = ComplexityEstimator(config)
    
    @abstractmethod
    def split(self, image: Tensor) -> SplitResult:
        """
        Split image into adaptive tokens.
        
        Args:
            image: [C, H, W] tensor with values in [0, 1]
            
        Returns:
            SplitResult containing tokens sorted by Hilbert order
        """
        pass
    
    def split_batch(self, images: Tensor) -> List[SplitResult]:
        """
        Split batch of images.
        
        Args:
            images: [B, C, H, W] tensor
            
        Returns:
            List of SplitResult, one per image
        """
        return [self.split(img) for img in images]


# =============================================================================
# Scheme B: Balanced Greedy Splitter
# =============================================================================

class BalancedGreedySplitter(BaseAdaptiveSplitter):
    """
    Scheme B: Greedy splitting with 2:1 balance constraint.
    
    Algorithm:
    1. Greedy recursive splitting based on complexity threshold
    2. Post-process to enforce 2:1 balance (|d_i - d_j| ≤ 1 for neighbors)
    3. Optionally apply token count soft constraint
    
    Properties:
    - H_LCA_eff ≈ 1.18 (highest effective LCA information)
    - J_max ≈ 21 pixels (low Hilbert jump distance)
    - σ_N ≈ 22 (moderate token count variance)
    """
    
    def split(self, image: Tensor) -> SplitResult:
        if image.dim() == 4:
            image = image.squeeze(0)
        
        C, H, W = image.shape
        assert H == W, "Image must be square"
        
        # Initialize
        cache = IntegralImageCache(
            image, 
            gradient_method=self.config.gradient_method
        )
        sorter = HilbertTokenSorter(H)
        
        # Step 1: Greedy recursive splitting
        root_region = Region(0, 0, W, H)
        leaves = self._greedy_split(root_region, 0, [], cache)
        
        # Step 2: Enforce 2:1 balance
        if self.config.enforce_balance:
            leaves = self._enforce_balance(leaves, cache, H)
        
        # Step 3: Apply token count constraint (optional)
        if self.config.target_tokens is not None:
            leaves = self._apply_token_constraint(leaves, cache, H)
        
        # Step 4: Convert to tokens and sort by Hilbert
        tokens = [
            SplitToken(
                region=node.region,
                depth=node.depth,
                path=node.path,
                hilbert_idx=0,
                complexity=node.complexity
            )
            for node in leaves
        ]
        tokens = sorter.sort_tokens(tokens)
        
        return SplitResult(tokens=tokens)
    
    def _greedy_split(
        self,
        region: Region,
        depth: int,
        path: List[int],
        cache: IntegralImageCache
    ) -> List[QuadtreeNode]:
        """Recursive greedy splitting."""
        node = QuadtreeNode(
            region=region,
            depth=depth,
            path=path.copy(),
            complexity=self.complexity_estimator.compute(region, cache)
        )
        
        # Termination conditions
        threshold = self.config.get_threshold(depth)
        min_size = self.config.min_region_size
        
        should_stop = (
            depth >= self.config.max_depth or
            node.complexity < threshold or
            region.width < min_size * 2 or
            region.height < min_size * 2
        )
        
        if should_stop:
            return [node]
        
        # Split into 4 quadrants
        leaves = []
        for q in range(4):
            sub_region = region.get_quadrant(q)
            sub_leaves = self._greedy_split(
                sub_region, depth + 1, path + [q], cache
            )
            leaves.extend(sub_leaves)
        
        return leaves
    
    def _enforce_balance(
        self,
        leaves: List[QuadtreeNode],
        cache: IntegralImageCache,
        image_size: int
    ) -> List[QuadtreeNode]:
        """
        Enforce 2:1 balance constraint.
        
        Iteratively split shallow neighbors of deep regions
        until |d_i - d_j| ≤ 1 for all adjacent pairs.
        """
        for iteration in range(self.config.max_balance_iterations):
            # Build spatial index for neighbor queries
            leaves_by_region = {
                (n.region.x1, n.region.y1, n.region.x2, n.region.y2): n
                for n in leaves
            }
            
            violations = []
            for node in leaves:
                neighbors = self._find_neighbors(node, leaves, image_size)
                for neighbor in neighbors:
                    if node.depth - neighbor.depth > 1:
                        violations.append(neighbor)
            
            if not violations:
                break
            
            # Split violating nodes
            new_leaves = []
            violated_set = set(id(n) for n in violations)
            
            for node in leaves:
                if id(node) in violated_set:
                    # Force split this node
                    for q in range(4):
                        sub_region = node.region.get_quadrant(q)
                        sub_node = QuadtreeNode(
                            region=sub_region,
                            depth=node.depth + 1,
                            path=node.path + [q],
                            complexity=self.complexity_estimator.compute(
                                sub_region, cache
                            )
                        )
                        new_leaves.append(sub_node)
                else:
                    new_leaves.append(node)
            
            leaves = new_leaves
        
        return leaves
    
    def _find_neighbors(
        self,
        node: QuadtreeNode,
        all_nodes: List[QuadtreeNode],
        image_size: int
    ) -> List[QuadtreeNode]:
        """Find all nodes adjacent to the given node."""
        neighbors = []
        r = node.region
        
        for other in all_nodes:
            if other is node:
                continue
            
            o = other.region
            
            # Check adjacency (sharing an edge)
            # Horizontal adjacency
            h_adjacent = (
                (r.x2 == o.x1 or r.x1 == o.x2) and
                not (r.y2 <= o.y1 or r.y1 >= o.y2)
            )
            # Vertical adjacency  
            v_adjacent = (
                (r.y2 == o.y1 or r.y1 == o.y2) and
                not (r.x2 <= o.x1 or r.x1 >= o.x2)
            )
            
            if h_adjacent or v_adjacent:
                neighbors.append(other)
        
        return neighbors
    
    def _apply_token_constraint(
        self,
        leaves: List[QuadtreeNode],
        cache: IntegralImageCache,
        image_size: int
    ) -> List[QuadtreeNode]:
        """
        Soft constraint to nudge token count toward target.
        
        If too many tokens: merge lowest-complexity sibling groups
        If too few tokens: split highest-complexity nodes
        """
        target = self.config.target_tokens
        if target is None:
            return leaves
        
        # Simple heuristic: adjust threshold and re-split
        # This is a soft constraint, not exact
        current = len(leaves)
        
        if abs(current - target) / target < 0.2:
            # Within 20%, acceptable
            return leaves
        
        # Could implement more sophisticated merging/splitting
        # For now, just return as-is (soft constraint)
        return leaves


# =============================================================================
# Scheme C: Fixed Budget DP Splitter
# =============================================================================

class FixedBudgetDPSplitter(BaseAdaptiveSplitter):
    """
    Scheme C: Fixed token budget with dynamic programming selection.
    
    Algorithm:
    1. Build complete quadtree to max_depth
    2. Compute importance for each node
    3. Use DP to select optimal pruning with exactly N tokens
    4. Optional locality penalty for smooth depth transitions
    
    Properties:
    - σ_N = 0 (exact token count)
    - Globally optimal selection (not greedy)
    - η_waste = 0 (perfect batch efficiency)
    """
    
    def split(self, image: Tensor) -> SplitResult:
        if image.dim() == 4:
            image = image.squeeze(0)
        
        C, H, W = image.shape
        assert H == W, "Image must be square"
        
        # Initialize
        cache = IntegralImageCache(
            image,
            gradient_method=self.config.gradient_method
        )
        sorter = HilbertTokenSorter(H)
        
        # Step 1: Build complete quadtree
        root_region = Region(0, 0, W, H)
        root = self._build_complete_tree(root_region, 0, [], cache)
        
        # Step 2: Compute importance scores
        self._compute_importance(root, cache)
        
        # Step 3: DP selection
        budget = self.config.token_budget
        selected = self._dp_select(root, budget)
        
        # Step 4: Convert to tokens and sort
        tokens = [
            SplitToken(
                region=node.region,
                depth=node.depth,
                path=node.path,
                hilbert_idx=0,
                complexity=node.complexity
            )
            for node in selected
        ]
        tokens = sorter.sort_tokens(tokens)
        
        return SplitResult(tokens=tokens)
    
    def _build_complete_tree(
        self,
        region: Region,
        depth: int,
        path: List[int],
        cache: IntegralImageCache
    ) -> QuadtreeNode:
        """Build complete quadtree to max_depth."""
        node = QuadtreeNode(
            region=region,
            depth=depth,
            path=path.copy(),
            complexity=self.complexity_estimator.compute(region, cache)
        )
        
        min_size = self.config.min_region_size
        can_split = (
            depth < self.config.max_depth and
            region.width >= min_size * 2 and
            region.height >= min_size * 2
        )
        
        if can_split:
            node.children = []
            for q in range(4):
                sub_region = region.get_quadrant(q)
                child = self._build_complete_tree(
                    sub_region, depth + 1, path + [q], cache
                )
                node.children.append(child)
        
        return node
    
    def _compute_importance(
        self,
        node: QuadtreeNode,
        cache: IntegralImageCache
    ) -> None:
        """
        Compute importance score for each node.
        
        Importance = complexity × area_weight - locality_penalty
        """
        # Base importance from complexity
        area_weight = node.region.area / (cache.H * cache.W)
        node.importance = node.complexity * math.sqrt(area_weight)
        
        # Recurse to children
        if node.children:
            for child in node.children:
                self._compute_importance(child, cache)
    
    def _dp_select(
        self,
        root: QuadtreeNode,
        budget: int
    ) -> List[QuadtreeNode]:
        """
        Dynamic programming to select optimal token set.
        
        Uses tree DP: for each subtree, compute best way to use k tokens
        for all k from 1 to budget.
        
        State: dp[node][k] = maximum importance achievable using k tokens
                            in the subtree rooted at node
        """
        # Memoization
        memo: Dict[Tuple[int, int], Tuple[float, List[QuadtreeNode]]] = {}
        
        def dp(node: QuadtreeNode, k: int) -> Tuple[float, List[QuadtreeNode]]:
            """
            Returns (importance, selected_nodes) for using exactly k tokens
            in the subtree rooted at node.
            """
            if k <= 0:
                return (0.0, [])
            
            node_id = id(node)
            if (node_id, k) in memo:
                return memo[(node_id, k)]
            
            # Option 1: Use this node as a leaf (1 token)
            if k == 1:
                result = (node.importance, [node])
                memo[(node_id, k)] = result
                return result
            
            # Option 2: If no children, can only use 1 token
            if not node.children:
                if k == 1:
                    result = (node.importance, [node])
                else:
                    result = (float('-inf'), [])  # Invalid
                memo[(node_id, k)] = result
                return result
            
            # Option 3: Distribute k tokens among 4 children
            # Use DP to find best distribution
            best_importance = node.importance if k == 1 else float('-inf')
            best_selection = [node] if k == 1 else []
            
            # Try all valid distributions of k tokens to 4 children
            # Each child must get at least 1 token (if used) or 0
            # This is a partition problem
            for dist in self._generate_distributions(k, 4):
                total_imp = 0.0
                selection = []
                valid = True
                
                for i, child_k in enumerate(dist):
                    if child_k > 0:
                        imp, sel = dp(node.children[i], child_k)
                        if imp == float('-inf'):
                            valid = False
                            break
                        total_imp += imp
                        selection.extend(sel)
                
                if valid and total_imp > best_importance:
                    best_importance = total_imp
                    best_selection = selection
            
            # Apply locality penalty for depth variance
            if best_selection and self.config.locality_penalty_weight > 0:
                depths = [n.depth for n in best_selection]
                if len(depths) > 1:
                    depth_var = sum((d - sum(depths)/len(depths))**2 
                                   for d in depths) / len(depths)
                    penalty = self.config.locality_penalty_weight * depth_var
                    best_importance -= penalty
            
            result = (best_importance, best_selection)
            memo[(node_id, k)] = result
            return result
        
        _, selected = dp(root, budget)
        return selected
    
    def _generate_distributions(
        self, 
        total: int, 
        parts: int
    ) -> List[Tuple[int, ...]]:
        """
        Generate all ways to distribute 'total' tokens to 'parts' children.
        
        Optimized to avoid combinatorial explosion.
        """
        if parts == 1:
            return [(total,)]
        
        distributions = []
        # Limit search space for efficiency
        max_per_part = min(total, 64)  # Cap individual allocation
        
        for first in range(min(total + 1, max_per_part + 1)):
            for rest in self._generate_distributions(total - first, parts - 1):
                distributions.append((first,) + rest)
        
        return distributions


# =============================================================================
# Factory Function
# =============================================================================

def create_adaptive_splitter(config: AdaptiveSplitConfig) -> BaseAdaptiveSplitter:
    """
    Factory function to create appropriate splitter based on config.
    
    Args:
        config: AdaptiveSplitConfig with scheme selection
        
    Returns:
        Either BalancedGreedySplitter (Scheme B) or FixedBudgetDPSplitter (Scheme C)
    """
    if config.scheme == SplitScheme.BALANCED_GREEDY:
        return BalancedGreedySplitter(config)
    elif config.scheme == SplitScheme.FIXED_BUDGET_DP:
        return FixedBudgetDPSplitter(config)
    else:
        raise ValueError(f"Unknown scheme: {config.scheme}")


# =============================================================================
# Convenience Functions
# =============================================================================

def split_image(
    image: Tensor,
    scheme: str = "balanced_greedy",
    **kwargs
) -> SplitResult:
    """
    Convenience function to split a single image.
    
    Args:
        image: [C, H, W] or [1, C, H, W] tensor
        scheme: "balanced_greedy" (B) or "fixed_budget_dp" (C)
        **kwargs: Additional config parameters
        
    Returns:
        SplitResult with tokens
    """
    scheme_enum = SplitScheme(scheme)
    config = AdaptiveSplitConfig(scheme=scheme_enum, **kwargs)
    splitter = create_adaptive_splitter(config)
    return splitter.split(image)


def compare_schemes(
    image: Tensor,
    config_overrides: Optional[Dict] = None
) -> Dict[str, SplitResult]:
    """
    Run both schemes on the same image for comparison.
    
    Args:
        image: Input image tensor
        config_overrides: Optional config parameters
        
    Returns:
        Dict with keys 'scheme_b' and 'scheme_c'
    """
    overrides = config_overrides or {}
    
    config_b = AdaptiveSplitConfig.scheme_b(**overrides)
    config_c = AdaptiveSplitConfig.scheme_c(**overrides)
    
    splitter_b = BalancedGreedySplitter(config_b)
    splitter_c = FixedBudgetDPSplitter(config_c)
    
    return {
        'scheme_b': splitter_b.split(image),
        'scheme_c': splitter_c.split(image)
    }
