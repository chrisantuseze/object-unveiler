import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# Helpers: attention + logits
# ---------------------------

def reduce_attn_to_BxNxN(attn):
    """
    Accepts attn of shape:
      - B x N x N
      - B x L x N x N  (per-layer) -> returns mean across L
    Returns: B x N x N
    """
    attn = torch.as_tensor(attn)
    if attn.dim() == 4:  # B x L x N x N
        return attn.mean(dim=1)
    elif attn.dim() == 3:  # B x N x N
        return attn
    else:
        raise ValueError(f"Unsupported attn dim: {attn.shape}")

def object_attention_from_target(attn_BxNxN, target_index=0, object_token_indices=None):
    """
    attn_BxNxN: tensor (B, N, N) attention where rows=query token, cols=key token.
    We take the row for the target token -> attention to each token,
    then keep only the entries for object tokens.

    Returns: scores (B, num_objects)
    """
    B, N, N2 = attn_BxNxN.shape
    assert N == N2, "Attention must be square over tokens."

    if object_token_indices is None:
        # Default: everything except target is an object token
        object_token_indices = [i for i in range(N) if i != target_index]

    # Gather target->objects attention
    # shape: (B, num_objects)
    scores = attn_BxNxN[:, target_index, object_token_indices]
    return scores, object_token_indices

# ---------------------------
# Visualization
# ---------------------------

def overlay_attention_with_probs(
    scene_img,
    object_boxes,
    contrib_scores,
    probs,
    target_idx=None,
    selected_idx=None,
    save_path=None,
    alpha=0.40,
    prob_format="{:.2f}",
    cmap_name="jet",
    font=cv2.FONT_HERSHEY_SIMPLEX
):
    """
    Visualize:
      - object_boxes colored by contrib_scores (jet colormap)
      - prints final_probs on each bbox (numeric)
      - labels only the target and the selected boxes with 'Target(p)' and 'Selected(p)'
    
    Args:
      scene_img: torch.Tensor (C,H,W) or numpy (H,W,C) or (C,H,W), values in [0,1] or uint8
      object_boxes: list of (x1,y1,x2,y2), length M
      contrib_scores: (M,) attention->object contribution (any scale; will be normalized)
      final_probs: (M,) or (B,M) (will take first row if B>1)
      target_idx: optional int (index into boxes)
      selected_idx: optional int
      save_path: optional path to save figure
      alpha: overlay blend weight
    """
    # --- convert inputs to numpy ---
    img = scene_img
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.array(img)

    # handle C,H,W -> H,W,C
    if img.ndim == 3 and img.shape[0] in (1,3):
        img = np.transpose(img, (1,2,0))
    if img.dtype != np.uint8:
        # assume [0,1]
        img = (img * 255).astype(np.uint8)

    H, W = img.shape[:2]
    M = len(object_boxes)

    contrib = np.asarray(contrib_scores, dtype=np.float32).flatten()
    if contrib.shape[0] != M:
        raise ValueError(f"contrib_scores length {contrib.shape[0]} != number of boxes {M}")

    if probs.ndim == 2:  # B x M -> take first row
        if probs.shape[0] > 1:
            # warn user but proceed
            print(f"[overlay] Warning: final_probs has shape {probs.shape}, using row 0. Pass a single-row array to avoid this.")
        probs = probs[0]
    probs = np.asarray(probs, dtype=np.float32).flatten()
    if probs.shape[0] != M:
        raise ValueError(f"final_probs length {probs.shape[0]} != number of boxes {M}")

    # normalize contrib to [0,1]
    cmin, cmax = float(contrib.min()), float(contrib.max())
    denom = (cmax - cmin) if (cmax != cmin) else 1.0
    norms = (contrib - cmin) / denom

    # create overlay
    overlay = img.copy()
    cmap = plt.get_cmap(cmap_name)

    for i, box in enumerate(object_boxes):
        x1, y1, x2, y2 = map(int, box)
        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        rgba = cmap(float(norms[i]))  # (r,g,b,a) floats 0..1
        rgb = tuple(int(255*c) for c in rgba[:3])
        # OpenCV uses BGR
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, thickness=-1)

    blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    # draw outlines and text (only target and selected get label text)
    for i, box in enumerate(object_boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # outline color: target = blue, selected = green, otherwise white thin box
        if (target_idx is not None and i == target_idx) and (selected_idx is not None and i == selected_idx):
            outline = (0, 255, 0)  # green if both (chosen target)
            thickness = 3
        elif target_idx is not None and i == target_idx:
            outline = (0, 0, 255)  # red (BGR)
            thickness = 3
        elif selected_idx is not None and i == selected_idx:
            outline = (0, 255, 0)  # green
            thickness = 3
        else:
            outline = (255, 255, 255)
            thickness = 2
        cv2.rectangle(blended, (x1, y1), (x2, y2), outline, thickness=thickness)

        # prepare probability text
        prob_txt = prob_format.format(float(probs[i]))
        # if this is target or selected, prefix with label
        if target_idx is not None and i == target_idx and selected_idx is not None and i == selected_idx:
            label_txt = f"Target/Selected {prob_txt}"
        elif target_idx is not None and i == target_idx:
            label_txt = f"Target {prob_txt}"
        elif selected_idx is not None and i == selected_idx:
            label_txt = f"Selected {prob_txt}"
        else:
            label_txt = prob_txt  # only the probability for others

        # choose font size relative to box height
        box_h = max(12, (y2 - y1))
        font_scale = max(0.4, min(2.0, box_h / 80.0))
        thickness_text = max(1, int(round(font_scale * 2)))

        # determine text color (white or black) for readability based on background color
        # sample the fill color at box center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        sample_rgb = blended[cy, cx, :3]  # BGR
        # compute luminance
        lum = 0.2126 * sample_rgb[2] + 0.7152 * sample_rgb[1] + 0.0722 * sample_rgb[0]
        text_color = (0,0,0) if lum > 130 else (255,255,255)
        outline_color = (255,255,255) if text_color == (0,0,0) else (0,0,0)

        # put label: draw outline (thicker) then text
        text_pos = (x1 + 4, max(y1 + int(12*font_scale), y1 + 14))  # inside top-left
        cv2.putText(blended, label_txt, text_pos, font, font_scale, outline_color, thickness=thickness_text+2, lineType=cv2.LINE_AA)
        cv2.putText(blended, label_txt, text_pos, font, font_scale, text_color, thickness=thickness_text, lineType=cv2.LINE_AA)

    # display/save
    disp = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(disp)
    plt.axis('off')
    plt.title("SRE: object contributions (color) + final probs (text)")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.show()

def plot_dual_bars(attn_scores_1d, probs_vec, picked_idx=None, save_path=None,
                   title="Attention vs. Final Decision (Probs)"):
    """
    attn_scores_1d: (M,)
    probs_vec: (M,) OR (B,M) -> will pick row 0 by default
    """
    attn = np.asarray(attn_scores_1d, dtype=np.float32)
    probs = np.asarray(probs_vec, dtype=np.float32)

    if probs.ndim == 2:
        # pick first row (or you may pass a specific batch row)
        if probs.shape[0] == 1:
            probs = probs[0]
        else:
            # Ambiguity: take first row but warn user (print shape)
            print(f"[plot_dual_bars] Warning: probs has shape {probs.shape}; using row 0. "
                  "If this is wrong pass a single row (M,) or select batch_index explicitly.")
            probs = probs[0]

    if attn.shape[0] != probs.shape[0]:
        raise ValueError(f"Length mismatch: attn has length {attn.shape[0]} but probs has length {probs.shape[0]}.\n"
                         "Make sure object_token_indices and object_boxes match the logits order and length.")

    x = np.arange(len(attn))
    width = 0.35
    plt.figure(figsize=(8,4))
    plt.bar(x - width/2, attn, width, label="Attention")
    plt.bar(x + width/2, probs, width, label="Logits → Probs", alpha=0.8)
    if picked_idx is not None:
        plt.axvline(int(picked_idx), color='green', linestyle='--', linewidth=2, label='Selected Object')
    plt.xlabel("Object Index (same order as logits / boxes)")
    plt.ylabel("Score / Probability")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.show()

# ---------------------------
# End-to-end convenience API
# ---------------------------

def make_sre_figures(
    scene_img,                # (C,H,W) tensor/ndarray in [0,1]
    object_bboxes,             # list[(x1,y1,x2,y2)] length M (order == logits order)
    logits,                   # (B,M) tensor/ndarray
    valid_mask,               # (B,M) tensor/ndarray bool or {0,1}
    attention_weights,              # (B,N,N) or (B,L,N,N) tensor/ndarray over *tokens*
    target_token_index=4,     # index of target token among N tokens
    object_token_indices=None,# indices of the M object tokens among N tokens (same order as logits)
    batch_index=0,            # select which batch item to visualize
    save_prefix=None          # e.g., "figs/scene1"
):
    """
    Produces:
      1) Scene overlay colored by attention(target->object_token)
      2) Dual bar chart: attention vs final probs
    """
    # 1) reduce attention (handle BxLxNxN or BxNxN) and pick batch
    attn_BxNxN = reduce_attn_to_BxNxN(attention_weights)
    attn_BxNxN = attn_BxNxN.detach().cpu() if isinstance(attn_BxNxN, torch.Tensor) else attn_BxNxN
    attn_b = attn_BxNxN[batch_index]  # (N, N)

    # 2) map target->objects attention
    attn_scores_BxM, obj_tok_idx = object_attention_from_target(
        attn_BxNxN, target_index=target_token_index, object_token_indices=object_token_indices
    )
    attn_scores = attn_scores_BxM[batch_index].detach().cpu().numpy() if isinstance(attn_scores_BxM, torch.Tensor) else attn_scores_BxM[batch_index]

    # 3) compute final selection probs (masked softmax) and chosen idx

    print("object_bboxes.shape", np.array(object_bboxes).shape)
    print("logits.shape", logits.shape)
    print("valid_mask.shape", valid_mask.shape)

    print("object_bboxes", object_bboxes)

    # Squeeze valid_mask to (10,)
    mask = valid_mask.squeeze(0)   # shape (10,)

    # Select only valid bboxes and logits
    object_bboxes = object_bboxes[mask]         # shape (num_valid, 4)
    logits = logits[0][mask]             # shape (num_valid,)


    print("object_bboxes.shape", np.array(object_bboxes).shape)
    print("logits.shape", logits.shape)

    probs = torch.softmax(logits, dim=-1)
    print("probs.shape", probs.shape)
    chosen_idx = int(np.argmax(probs))

    # # 4) Visuals
    # overlay_attention_with_probs(
    #     scene_img=scene_img,
    #     object_boxes=object_bboxes,
    #     contrib_scores=attn_scores,
    #     probs=probs,
    #     target_idx=target_token_index if target_token_index < len(object_bboxes) else None,
    #     selected_idx=chosen_idx,
    #     save_path=f"{save_prefix}_attn_probs_overlay.png" if save_prefix else None,
    #     alpha=0.4,
    # )

    save2 = f"{save_prefix}_dual_bars.png" if save_prefix else None
    plot_dual_bars(attn_scores, probs, picked_idx=chosen_idx, save_path=save2)

    visualize_sre_decisions(
        rgb_image=scene_img,
        object_bboxes=object_bboxes,
        object_scores=probs.numpy(),#logits.numpy() if isinstance(logits, torch.Tensor) else logits,
        attention_weights=attn_scores,
        target_bbox=object_bboxes[target_token_index] if target_token_index < len(object_bboxes) else None,
        valid_mask=valid_mask,
        save_path=f"{save_prefix}_sre_visualization1.png" if save_prefix else None,
        selected_object_idx=chosen_idx
    )

    # print("object_bboxes", object_bboxes)

    # visualize_attention_on_objects(
    #     image_path=scene_img,
    #     bboxes=object_bboxes,
    #     attention_probs=logits.numpy() if isinstance(logits, torch.Tensor) else logits,
    #     target_obj_id=target_token_index if target_token_index < len(object_bboxes) else -1,
    #     selected_obj_id=chosen_idx,
    #     save_path=f"{save_prefix}_sre_visualization2.png" if save_prefix else None,
    #     figsize=(12, 8),
    #     alpha=0.7
    # )


def do_analysis(test_scene, save_dir="./figures/"):
    """
    Generate attention visualization for a batch of test data.
    
    Args:
        model: Your trained SpatialEncoder model
        batch_data: Dictionary containing:
            - scene_image: [B, C, H, W]
            - target_mask: [B, C, H, W]  
            - object_masks: [B, N, C, H, W]
            - bboxes: [B, N, 4]
            - rgb_images: [B, H, W, 3] (for visualization)
        save_dir: Directory to save visualizations
    """
        
    make_sre_figures(
        scene_img=test_scene['rgb'],
        object_bboxes=test_scene['object_bboxes'],
        logits=test_scene['obstacle_scores'],
        attention_weights=test_scene['attention_weights'],
        valid_mask=test_scene['valid_mask'],
        save_prefix="sre_analysis"
    )
    
    # Disable attention saving
    # model.save_attention = False
    
    print(f"Generated attention visualizations in {save_dir}")


from matplotlib.patches import Rectangle

def visualize_sre_decisions(rgb_image, object_bboxes, object_scores, attention_weights, 
                           target_bbox=None, valid_mask=None, save_path=None, 
                           selected_object_idx=None):
    """
    Visualize SRE attention maps and decision scores.
    
    Args:
        rgb_image: RGB image of the scene (H, W, 3) - numpy array
        object_bboxes: List/tensor of bounding boxes [(x1,y1,x2,y2), ...] - shape [N, 4]
        object_scores: SRE output scores for each object [N] - higher = more likely to be selected
        attention_weights: Attention weights from transformer [B, N, N] or [B*heads, N, N]
        target_bbox: Target object bounding box (x1,y1,x2,y2)
        valid_mask: Boolean mask indicating which objects are real (not padding) [N]
        save_path: Path to save the visualization
        selected_object_idx: Index of the object actually selected by the model
    """
    
    # fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    
    # 1. Original scene with object boxes and scores
    axes.imshow(rgb_image)
    axes.set_title('Scene with SRE Obstacle Scores', fontsize=14, fontweight='bold', loc='left')
    
    # Sort objects by score for better visualization
    sorted_indices = np.argsort(object_scores)[::-1]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(object_bboxes)))
    
    for rank, idx in enumerate(sorted_indices):
        bbox = object_bboxes[idx]
        score = object_scores[idx]
        x1, y1, x2, y2 = bbox
        
        # Thicker border for higher-ranked objects
        linewidth = 4 if rank < 3 else 2
        
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=linewidth, edgecolor=colors[rank], facecolor='none')
        axes.add_patch(rect)
        
        # Score annotation
        axes.text(x1, y1-6, f'#{rank+1}: {score:.3f}', 
                    fontsize=11, color='black', fontweight='bold', #colors[rank]
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[rank], alpha=0.9))
    
    # Highlight target if provided
    if target_bbox is not None:
        x1, y1, x2, y2 = target_bbox
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=4, edgecolor='red', facecolor='none', linestyle='--')
        axes.add_patch(rect)
        axes.text(x1, y1-17, 'TARGET', fontsize=12, color='red', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.9))
    
    # Highlight selected object if provided
    if selected_object_idx is not None and selected_object_idx < len(object_bboxes):
        bbox = object_bboxes[selected_object_idx]
        x1, y1, x2, y2 = bbox
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=5, edgecolor='lime', facecolor='none', linestyle='-')
        axes.add_patch(rect)
        axes.text(x2, y1-6, 'SELECTED', fontsize=10, color='lime', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
    
    axes.axis('off')
    
    # # 2. Attention-based visualization
    # axes[1].imshow(rgb_image)
    # axes[1].set_title('SRE Attention Weights', fontsize=14, fontweight='bold')
    
    # # Create heatmap overlay based on attention
    # heatmap = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
    
    # for i, (bbox, attention) in enumerate(zip(object_bboxes, attention_weights)):
    #     x1, y1, x2, y2 = bbox.astype(int)
    #     # Ensure bbox is within image bounds
    #     x1, y1 = max(0, x1), max(0, y1)
    #     x2, y2 = min(rgb_image.shape[1], x2), min(rgb_image.shape[0], y2)
        
    #     # Fill bounding box area with attention weight
    #     heatmap[y1:y2, x1:x2] = attention
        
    #     # Draw bounding box with attention-based color intensity
    #     color_intensity = max(0.3, attention)  # Ensure minimum visibility
    #     color = plt.cm.Reds(color_intensity)
        
    #     rect = Rectangle((x1, y1), x2-x1, y2-y1, 
    #                     linewidth=3, edgecolor=color, facecolor='none')
    #     axes[1].add_patch(rect)
        
    #     # Add attention score text
    #     axes[1].text(x1+5, y1+15, f'{attention:.3f}', 
    #                 fontsize=10, color='white', fontweight='bold',
    #                 bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    
    # # Overlay heatmap with transparency
    # axes[1].imshow(heatmap, alpha=0.4, cmap='Reds', vmin=0, vmax=1)
    # axes[1].axis('off')
    
    # # 3. Score vs Attention comparison
    # axes[2].scatter(attention_weights, object_scores, 
    #                c=range(len(object_scores)), cmap='viridis', s=100, alpha=0.7)
    
    # # Add object labels
    # for i, (att, score) in enumerate(zip(attention_weights, object_scores)):
    #     axes[2].annotate(f'Obj{i}', 
    #                     (att, score), xytext=(5, 5), textcoords='offset points',
    #                     fontsize=9, alpha=0.8)
    
    # axes[2].set_xlabel('Attention Weight', fontsize=12)
    # axes[2].set_ylabel('SRE Score', fontsize=12)
    # axes[2].set_title('Attention vs Score Correlation', fontsize=14, fontweight='bold')
    # axes[2].grid(True, alpha=0.3)
    
    # # Add correlation coefficient
    # if len(attention_weights) > 1:
    #     corr = np.corrcoef(attention_weights, object_scores)[0, 1]
    #     axes[2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
    #                 transform=axes[2].transAxes, fontsize=11,
    #                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

def visualize_attention_on_objects(image_path, bboxes, attention_probs, target_obj_id, selected_obj_id, 
                                 save_path=None, figsize=(12, 8), alpha=0.7):
    """
    Visualize attention probabilities overlaid on objects in a scene image.
    
    Args:
        image_path (str): Path to the RGB scene image
        bboxes (np.ndarray): Nx4 array of bounding boxes [x1, y1, x2, y2]
        attention_probs (np.ndarray): 1xN array of attention probabilities
        target_obj_id (int): Index of the target object
        selected_obj_id (int): Index of the selected object
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size for matplotlib
        alpha (float): Transparency for attention overlay
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    
    # Load and display the image
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path  # Assume it's already a PIL Image or numpy array
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    
    # Flatten attention probabilities if needed
    if attention_probs.ndim > 1:
        attention_probs = attention_probs.flatten()
    
    # Normalize attention probabilities for color mapping
    norm_probs = (attention_probs - attention_probs.min()) / (attention_probs.max() - attention_probs.min())
    
    # Color maps for attention (warmer colors = higher attention)
    cmap = plt.cm.Reds
    
    # Draw bounding boxes with attention overlays
    for i, (bbox, prob, norm_prob) in enumerate(zip(bboxes, attention_probs, norm_probs)):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Choose colors and styles based on object type
        if i == target_obj_id:
            edge_color = 'blue'
            edge_width = 4
            label = f'Target (ID:{i})'
            label_color = 'blue'
        elif i == selected_obj_id:
            edge_color = 'green'
            edge_width = 4
            label = f'Selected (ID:{i})'
            label_color = 'green'
        else:
            edge_color = 'white'
            edge_width = 2
            label = f'ID:{i}'
            label_color = 'white'
        
        # Create attention overlay (filled rectangle with alpha)
        attention_color = cmap(norm_prob)
        attention_rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=0,
            facecolor=attention_color,
            alpha=alpha
        )
        ax.add_patch(attention_rect)
        
        # Create bounding box outline
        bbox_rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=edge_width,
            edgecolor=edge_color,
            facecolor='none'
        )
        ax.add_patch(bbox_rect)
        
        # Add attention probability text
        prob_text = f'{prob:.3f}'
        ax.text(x1 + 5, y1 + 15, prob_text, 
                fontsize=10, fontweight='bold',
                color='white', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # Add object label
        ax.text(x1 + 5, y2 - 5, label,
                fontsize=12, fontweight='bold',
                color=label_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Create colorbar for attention probabilities
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=attention_probs.min(), vmax=attention_probs.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Probability', rotation=270, labelpad=20, fontsize=12)
    
    # Create legend
    legend_elements = [
        patches.Patch(color='blue', label='Target Object'),
        patches.Patch(color='green', label='Selected Object'),
        patches.Patch(color='white', label='Other Objects')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Set title and remove axes
    ax.set_title('Object Attention Visualization', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig