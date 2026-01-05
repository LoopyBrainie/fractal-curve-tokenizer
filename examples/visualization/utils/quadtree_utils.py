import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Patch:
    x: int
    y: int
    size: int
    depth: int
    complexity: float

class QuadtreeVisualizer:
    def __init__(self, 
                 alpha: float = 0.5, 
                 sigma_0_sq: float = 0.01, 
                 g_0_sq: float = 0.08,
                 min_size: int = 4,
                 max_depth: int = 6):
        self.alpha = alpha
        self.sigma_0_sq = sigma_0_sq
        self.g_0_sq = g_0_sq
        self.min_size = min_size
        self.max_depth = max_depth

    def calculate_complexity(self, img_region: np.ndarray) -> float:
        """
        Calculate complexity C(R) = alpha * C_var(R) + (1-alpha) * C_grad(R)
        """
        if img_region.size == 0:
            return 0.0
            
        # Normalize image to [0, 1] if not already
        if img_region.dtype == np.uint8:
            region = img_region.astype(float) / 255.0
        else:
            region = img_region

        # Variance component
        var = np.var(region)
        c_var = var / (var + self.sigma_0_sq)

        # Gradient component
        # Simple Sobel gradient
        gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        grad_energy = np.mean(gx**2 + gy**2)
        c_grad = grad_energy / (grad_energy + self.g_0_sq)

        return self.alpha * c_var + (1 - self.alpha) * c_grad

    def split(self, image: np.ndarray) -> List[Patch]:
        """
        Recursively split the image into quadtree patches.
        """
        h, w = image.shape[:2]
        # Ensure square image for simplicity in visualization
        size = min(h, w)
        # Crop to center square
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        img_square = image[start_y:start_y+size, start_x:start_x+size]
        
        if len(img_square.shape) == 3:
            img_gray = cv2.cvtColor(img_square, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_square

        patches = []
        
        def recursive_split(x, y, s, depth):
            region = img_gray[y:y+s, x:x+s]
            complexity = self.calculate_complexity(region)
            
            # Threshold logic: tau_d = tau_0 * gamma^d
            # Simplified for visualization: split if complexity is high and not too deep
            # Using a fixed threshold for demo purposes, or the one from the paper
            tau = 0.3 # Example threshold
            
            if depth < self.max_depth and s > self.min_size and complexity > tau:
                half = s // 2
                recursive_split(x, y, half, depth + 1)
                recursive_split(x + half, y, half, depth + 1)
                recursive_split(x, y + half, half, depth + 1)
                recursive_split(x + half, y + half, half, depth + 1)
            else:
                patches.append(Patch(x + start_x, y + start_y, s, depth, complexity))

        recursive_split(0, 0, size, 0)
        return patches
