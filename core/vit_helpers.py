"""Utilities for downloading a lightweight ViT from HuggingFace and visualizing attention maps.

Functions:
- load_vit_model: download/load model + image processor and move to device
- get_attention_overlays: compute cls-token attention maps per layer and return overlay images
"""
from typing import Tuple, List, Optional
import os
import numpy as np
from PIL import Image
import torch

try:
    # modern transformers
    from transformers import ViTModel, AutoImageProcessor
except Exception:
    # fallback names for older transformers
    from transformers import ViTModel
    try:
        from transformers import AutoFeatureExtractor as AutoImageProcessor
    except Exception:
        AutoImageProcessor = None


def _ensure_cache_dir(root: str = ".cache/vit_models") -> str:
    os.makedirs(root, exist_ok=True)
    return root


def load_vit_model(model_name: str = "google/vit-small-patch16-224", cache_root: Optional[str] = None, device: Optional[torch.device] = None) -> Tuple[ViTModel, object]:
    """Download (if needed) and load a ViT model and its image processor.

    Returns (model, processor). Model is moved to `device` if provided.
    """
    cache_root = _ensure_cache_dir(cache_root or ".cache/vit_models")
    cache_dir = os.path.join(cache_root, model_name.replace('/', '_'))
    os.makedirs(cache_dir, exist_ok=True)

    # Load processor / feature extractor
    if AutoImageProcessor is None:
        raise RuntimeError("transformers AutoImageProcessor / AutoFeatureExtractor not available in this environment")

    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    # Load model with attention outputs
    model = ViTModel.from_pretrained(model_name, output_attentions=True, cache_dir=cache_dir)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    return model, processor


def _create_overlay(original_pil: Image.Image, attn_map: np.ndarray, cmap='jet', alpha: float = 0.5) -> Image.Image:
    """Create a heatmap overlay on the original PIL image from a 2D attention map (0..1).

    attn_map is expected to be a 2D numpy array normalized between 0 and 1.
    """
    import matplotlib.cm as cm

    # Normalize
    m = attn_map
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)

    # Apply colormap -> RGBA floats
    cmap_fn = cm.get_cmap(cmap)
    colored = cmap_fn(m)[:, :, :3]  # drop alpha channel

    # Resize heatmap to original image size
    heatmap = Image.fromarray((colored * 255).astype(np.uint8)).resize(original_pil.size, resample=Image.BILINEAR)

    heatmap = heatmap.convert('RGBA')
    base = original_pil.convert('RGBA')

    # Blend
    blended = Image.blend(base, heatmap, alpha=alpha)
    return blended.convert('RGB')


def get_attention_overlays(pil_image: Image.Image, model: ViTModel, processor, device: Optional[torch.device] = None, max_layers: Optional[int] = None, alpha: float = 0.6) -> List[Image.Image]:
    """Compute attention overlays for each transformer encoder layer.

    Returns a list of PIL.Image overlays (one per layer). The attention maps are CLS->patch attentions
    averaged over heads and upsampled to the original image size.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare image for model
    # The processor handles resizing/cropping to model expected size
    inputs = processor(images=pil_image, return_tensors='pt')
    pixel_values = inputs.get('pixel_values')
    if pixel_values is None:
        raise RuntimeError('Processor did not return pixel_values')

    pixel_values = pixel_values.to(device)

    # Forward with attentions
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    attentions = outputs.attentions  # tuple: (layer_count,) each tensor shape (batch, heads, seq_len, seq_len)

    # Determine patch grid size: seq_len = 1 + num_patches
    # num_patches = (image_size // patch_size) ** 2
    try:
        patch_size = getattr(model.config, 'patch_size', None)
        model_img_size = getattr(model.config, 'image_size', None)
    except Exception:
        patch_size = None
        model_img_size = None

    # If processor resizes, use its size config
    try:
        proc_size = processor.size if hasattr(processor, 'size') else None
        if isinstance(proc_size, dict):
            proc_size = proc_size.get('shortest_edge') or proc_size.get('height') or proc_size.get('width')
    except Exception:
        proc_size = None

    # pick an image size used by the processor/model
    target_image_size = None
    for s in (proc_size, model_img_size, 224):
        if s:
            target_image_size = int(s)
            break

    # infer grid
    inferred_patch = int(patch_size) if patch_size else 16
    grid_size = target_image_size // inferred_patch

    overlays: List[Image.Image] = []

    layer_count = len(attentions)
    if max_layers is None:
        max_layers = layer_count
    max_layers = min(max_layers, layer_count)

    # We take batch 0
    for layer_idx in range(max_layers):
        attn = attentions[layer_idx]  # (batch, heads, seq_len, seq_len)
        attn = attn[0]  # remove batch -> (heads, seq_len, seq_len)

        # CLS token is index 0; take attention from CLS to all patches (exclude CLS)
        # attn[:, 0, 1:] -> (heads, num_patches)
        cls_attn = attn[:, 0, 1:]

        # Average over heads
        avg_attn = cls_attn.mean(axis=0).cpu().numpy()  # (num_patches,)

        # reshape to grid
        try:
            grid_h = grid_w = int(np.sqrt(avg_attn.shape[0]))
        except Exception:
            grid_h = grid_w = grid_size

        attn_grid = avg_attn.reshape(grid_h, grid_w)

        # Upsample to original image size and create overlay
        # Normalize between 0 and 1
        attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-12)

        # Create overlay image
        overlay = _create_overlay(pil_image, attn_grid, alpha=alpha)
        overlays.append(overlay)

    return overlays


def get_attention_overlays_per_head(pil_image: Image.Image, model: ViTModel, processor, layer_idx: int = 0, device: Optional[torch.device] = None, alpha: float = 0.6) -> List[Image.Image]:
    """Compute attention overlays for each head in a specific layer.

    Returns a list of PIL.Image overlays (one per head). Shows CLS->patch attention for each head separately.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare image for model
    inputs = processor(images=pil_image, return_tensors='pt')
    pixel_values = inputs.get('pixel_values')
    if pixel_values is None:
        raise RuntimeError('Processor did not return pixel_values')

    pixel_values = pixel_values.to(device)

    # Forward with attentions
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    attentions = outputs.attentions  # tuple: (layer_count,) each tensor shape (batch, heads, seq_len, seq_len)

    # Determine patch grid size
    try:
        patch_size = getattr(model.config, 'patch_size', None)
        model_img_size = getattr(model.config, 'image_size', None)
    except Exception:
        patch_size = None
        model_img_size = None

    try:
        proc_size = processor.size if hasattr(processor, 'size') else None
        if isinstance(proc_size, dict):
            proc_size = proc_size.get('shortest_edge') or proc_size.get('height') or proc_size.get('width')
    except Exception:
        proc_size = None

    target_image_size = None
    for s in (proc_size, model_img_size, 224):
        if s:
            target_image_size = int(s)
            break

    inferred_patch = int(patch_size) if patch_size else 16
    grid_size = target_image_size // inferred_patch

    overlays: List[Image.Image] = []

    if layer_idx >= len(attentions):
        layer_idx = len(attentions) - 1

    attn = attentions[layer_idx]  # (batch, heads, seq_len, seq_len)
    attn = attn[0]  # remove batch -> (heads, seq_len, seq_len)

    num_heads = attn.shape[0]

    # Process each head separately
    for head_idx in range(num_heads):
        # CLS token is index 0; take attention from CLS to all patches (exclude CLS)
        cls_attn = attn[head_idx, 0, 1:]  # (num_patches,)

        # reshape to grid
        try:
            grid_h = grid_w = int(np.sqrt(cls_attn.shape[0]))
        except Exception:
            grid_h = grid_w = grid_size

        attn_grid = cls_attn.cpu().numpy().reshape(grid_h, grid_w)

        # Normalize between 0 and 1
        attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-12)

        # Create overlay image
        overlay = _create_overlay(pil_image, attn_grid, alpha=alpha)
        overlays.append(overlay)

    return overlays


def get_attention_rollout(pil_image: Image.Image, model: ViTModel, processor, device: Optional[torch.device] = None, alpha: float = 0.6, discard_ratio: float = 0.9) -> Image.Image:
    """Compute attention rollout visualization (Abnar & Zuidema 2020).

    Attention rollout follows the attention flow from the classification token through all layers.
    Returns a single PIL.Image overlay showing the accumulated attention.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare image for model
    inputs = processor(images=pil_image, return_tensors='pt')
    pixel_values = inputs.get('pixel_values')
    if pixel_values is None:
        raise RuntimeError('Processor did not return pixel_values')

    pixel_values = pixel_values.to(device)

    # Forward with attentions
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    attentions = outputs.attentions  # tuple: (layer_count,) each tensor shape (batch, heads, seq_len, seq_len)

    # Determine patch grid size
    try:
        patch_size = getattr(model.config, 'patch_size', None)
        model_img_size = getattr(model.config, 'image_size', None)
    except Exception:
        patch_size = None
        model_img_size = None

    try:
        proc_size = processor.size if hasattr(processor, 'size') else None
        if isinstance(proc_size, dict):
            proc_size = proc_size.get('shortest_edge') or proc_size.get('height') or proc_size.get('width')
    except Exception:
        proc_size = None

    target_image_size = None
    for s in (proc_size, model_img_size, 224):
        if s:
            target_image_size = int(s)
            break

    inferred_patch = int(patch_size) if patch_size else 16
    grid_size = target_image_size // inferred_patch

    # Convert to numpy and average over heads
    result = None
    for layer_idx, attn in enumerate(attentions):
        attn = attn[0]  # remove batch -> (heads, seq_len, seq_len)
        
        # Average over heads
        attn = attn.mean(axis=0).cpu().numpy()  # (seq_len, seq_len)
        
        # Add residual connection (identity matrix)
        seq_len = attn.shape[0]
        attn = attn + np.eye(seq_len)
        
        # Re-normalize
        attn = attn / attn.sum(axis=-1, keepdims=True)
        
        if result is None:
            result = attn
        else:
            # Matrix multiplication for rollout
            result = np.matmul(attn, result)
    
    # Take CLS token attention to patches
    cls_attn = result[0, 1:]  # (num_patches,)
    
    # Apply discard ratio (keep top attention values)
    flat_attn = cls_attn.flatten()
    threshold = np.percentile(flat_attn, discard_ratio * 100)
    cls_attn = np.where(cls_attn < threshold, 0, cls_attn)
    
    # reshape to grid
    try:
        grid_h = grid_w = int(np.sqrt(cls_attn.shape[0]))
    except Exception:
        grid_h = grid_w = grid_size

    attn_grid = cls_attn.reshape(grid_h, grid_w)

    # Normalize between 0 and 1
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-12)

    # Create overlay image
    overlay = _create_overlay(pil_image, attn_grid, alpha=alpha)
    return overlay
