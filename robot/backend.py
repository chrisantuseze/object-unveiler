"""Object-selection backend for the real-robot server: segmentation + one selector per method.

Every method answers the same question: which segmented object should be removed next to reach the target? The
answer is an index into the objects the server segmented, so all methods share the Jetson's executor and differ only
in the choice.

    sre        SRE after PPO fine-tuning (the full Unveiler reasoning module)
    sre_il     SRE after imitation learning only
    heuristic  the periphery + target-distance heuristic that supervised the SRE (policy/grasping.py)
    gpt4o      GPT-4o obstacle selection (baseline/gpt.py, as in the sim tables)
    clip       zero-shot CLIP obstacle selection (baseline/clip_eval.py, as in Table IV)
    random     uniform over the selectable objects

Preprocessing matches the sim evaluation (eval_agent.py -> Policy.exploit_unveiler_rl): the (optionally cropped)
frame is resized to the 400x400 sim resolution and segmented by the same Mask R-CNN, and the SRE gets 100x100 object
masks, a 100x100 target mask and a grey-scale 100x100 scene image, padded to ``num_patches`` object slots.
"""

import json
import logging
import os
import tempfile
import time
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

import policy.grasping as grasping
import utils.general_utils as general_utils
from mask_rg.object_segmenter import ObjectSegmenter
from policy.sre_actor_critic import SREActorCritic
from policy.sre_model import SpatialEncoder
from robot.protocol import METHODS, encode_image, encode_mask

logger = logging.getLogger(__name__)

SIM_SIZE = 400        # sim camera resolution the segmenter and the SRE were trained on
SRE_SIZE = 100        # SRE input resolution (general_utils.resize_mask)
TARGET_MATCH_IOU = 0.3


class UnveilerBackend:
    def __init__(self, device: str = "cuda", sre_rl_ckpt: str = "save/sre_rl/sre_rl_best.pt",
                 sre_il_ckpt: str = "save/sre/sre_model_best.pt", num_patches: int = 10,
                 seg_threshold: float = 0.97, crop: Optional[Sequence[int]] = None,
                 output_dir: str = "save/real_eval", seed: int = 0):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_patches = num_patches
        self.crop = list(crop) if crop else None
        self.output_dir = Path(output_dir)
        self.rng = np.random.RandomState(seed)
        model_args = Namespace(device=self.device, num_patches=num_patches, sequence_length=1)

        logger.info("Loading Mask R-CNN (threshold %.2f) ...", seg_threshold)
        self.segmenter = ObjectSegmenter(model_args)
        self.segmenter.threshold = seg_threshold
        self._seg_dir = tempfile.mkdtemp(prefix="unveiler_seg_")   # from_maskrcnn writes debug PNGs here

        logger.info("Loading SRE-RL from %s ...", sre_rl_ckpt)
        self.sre_rl = SREActorCritic(model_args).to(self.device)
        ckpt = torch.load(sre_rl_ckpt, map_location=self.device, weights_only=False)
        self.sre_rl.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        self.sre_rl.eval()

        logger.info("Loading SRE-IL from %s ...", sre_il_ckpt)
        self.sre_il = SpatialEncoder(model_args).to(self.device)
        self.sre_il.load_state_dict(torch.load(sre_il_ckpt, map_location=self.device))
        self.sre_il.eval()

        self._gpt = None    # loaded on first use: needs OPENAI_API_KEY
        self._clip = None   # loaded on first use: ~350 MB of GPU memory
        self.log_dir = self.output_dir / "default" / "episode_0000"
        self._warmup()

    @torch.no_grad()
    def _warmup(self):
        """One pass through every network so the first real `select` reports warm latency (CUDA init, cuDNN autotune)."""
        blank = np.zeros((SIM_SIZE, SIM_SIZE, 3), np.uint8)
        self.segmenter.from_maskrcnn(blank, dir=self._seg_dir, bbox=True, dim=(SIM_SIZE, SIM_SIZE))
        box = np.zeros((SIM_SIZE, SIM_SIZE), np.uint8)
        box[150:250, 150:250] = 255
        inputs = self._sre_inputs(blank, box, [box], [[150.0, 150.0, 250.0, 250.0]])
        self.sre_rl(*inputs)
        self.sre_il(*inputs)

    @classmethod
    def from_args(cls, args):
        return cls(device=args.device, sre_rl_ckpt=args.sre_rl_ckpt, sre_il_ckpt=args.sre_il_ckpt,
                   num_patches=args.num_patches, seg_threshold=args.seg_threshold, crop=args.crop,
                   output_dir=args.output_dir, seed=args.seed)

    @property
    def settings(self) -> dict:
        return {"methods": list(METHODS), "default_method": "sre", "num_patches": self.num_patches,
                "sim_image_size": SIM_SIZE, "crop": self.crop, "seg_threshold": self.segmenter.threshold}

    # ── ops ──────────────────────────────────────────────────────────────────────────────────────────────────────

    def reset(self, session: str, episode) -> dict:
        self.log_dir = self.output_dir / str(session) / f"episode_{int(episode):04d}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return {"log_dir": str(self.log_dir)}

    def select(self, image_bgr: np.ndarray, target_mask: np.ndarray, method: str = "sre",
               reachable_mask: Optional[np.ndarray] = None, target_visible: Optional[bool] = None,
               step: Optional[int] = None, return_overlay: bool = False, decode_ms: float = 0.0) -> dict:
        if method not in METHODS:
            raise ValueError(f"unknown method '{method}', expected one of {METHODS}")
        if target_mask.shape[:2] != image_bgr.shape[:2]:
            raise ValueError(f"target_mask {target_mask.shape[:2]} and image {image_bgr.shape[:2]} differ in size")
        if reachable_mask is not None and reachable_mask.shape[:2] != image_bgr.shape[:2]:
            raise ValueError(f"reachable_mask {reachable_mask.shape[:2]} and image {image_bgr.shape[:2]} differ in size")
        t0 = time.perf_counter()

        # ── segment at sim resolution ──
        x0, y0, x1, y1 = self._roi(image_bgr.shape)
        color = cv2.resize(cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), (SIM_SIZE, SIM_SIZE))
        target = self._to_sim(target_mask, (x0, y0, x1, y1))
        masks, _, _, bboxes = self.segmenter.from_maskrcnn(color, dir=self._seg_dir, bbox=True,
                                                           dim=(SIM_SIZE, SIM_SIZE))
        t_seg = time.perf_counter()

        n = len(masks)
        full_masks = [self._to_frame(m, image_bgr.shape, (x0, y0, x1, y1)) for m in masks]
        centroids = [_centroid(m) for m in full_masks]
        if reachable_mask is None:
            reachable = [True] * n
        else:
            reachable = [bool(reachable_mask[cy, cx] > 0) for cx, cy in centroids]

        target_index = -1 if target_visible is False else _match_target(target, masks)
        reply = {
            "method": method, "num_objects": n, "truncated": n > self.num_patches,
            "target_visible": target_index >= 0, "target_index": target_index,
            "target_reachable": bool(target_index < 0 or reachable[target_index]),
            "objects": [{"index": i, "centroid": centroids[i],
                         "bbox": self._bbox_to_frame(bboxes[i], (x0, y0, x1, y1)),
                         "area": int(np.count_nonzero(full_masks[i])), "reachable": reachable[i]}
                        for i in range(n)],
            "unfiltered_index": None, "probs": None, "reason": "",
        }

        # ── choose ──
        if n == 0:
            chosen, reply["reason"] = -1, "no objects segmented"
        elif target.max() == 0:
            chosen, reply["reason"] = -1, "empty target mask (after crop)"
        else:
            candidates = [i for i in range(n) if reachable[i]]
            if not candidates:
                chosen, reply["reason"] = -1, "no reachable object"
            elif method in ("sre", "sre_il"):
                chosen = self._select_sre(method, color, target, masks, bboxes, reachable, reply)
            elif method == "heuristic":
                chosen = self._select_heuristic(target, masks, target_index, reachable)
            elif method == "gpt4o":
                chosen = self._select_gpt(target, masks, reachable)
            elif method == "clip":
                chosen = self._select_clip(target, masks, candidates)
            else:
                chosen = int(self.rng.choice(candidates))
        t_sel = time.perf_counter()

        if chosen >= n:   # the SRE / GPT-4o convention: an index past the object list means "the target itself"
            if target_index >= 0:
                chosen = target_index
            else:
                chosen, reply["reason"] = -1, "method chose the target, but the target is not visible"
        reply.update({
            "chosen_index": chosen,
            "is_target": bool(chosen >= 0 and chosen == target_index),
            "chosen_reachable": bool(chosen >= 0 and reachable[chosen]),
            "chosen_centroid": centroids[chosen] if chosen >= 0 else None,
            "chosen_bbox": reply["objects"][chosen]["bbox"] if chosen >= 0 else None,
            "chosen_mask": encode_mask(full_masks[chosen]) if chosen >= 0 else None,
            "timing_ms": {"decode": round(decode_ms, 1), "segment": round(1e3 * (t_seg - t0), 1),
                          "select": round(1e3 * (t_sel - t_seg), 1),
                          "total": round(decode_ms + 1e3 * (t_sel - t0), 1)},
        })

        overlay = draw_overlay(image_bgr, full_masks, target_mask, chosen, target_index, reachable)
        reply["log_dir"] = self._log_step(step, method, image_bgr, target_mask, overlay, masks, reply)
        if return_overlay:
            reply["overlay"] = encode_image(overlay, quality=85)
        return reply

    # ── selectors ────────────────────────────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _select_sre(self, method, color, target, masks, bboxes, reachable, reply) -> int:
        scene_t, target_t, masks_t, bboxes_t = self._sre_inputs(color, target, masks, bboxes)
        if method == "sre":
            logits, _, _ = self.sre_rl(scene_t, target_t, masks_t, bboxes_t)
        else:
            logits, _ = self.sre_il(scene_t, target_t, masks_t, bboxes_t)
        logits = logits[0].float().clone()
        k = min(len(masks), self.num_patches)
        reply["unfiltered_index"] = int(torch.argmax(logits).item())
        reply["probs"] = [round(p, 4) for p in torch.softmax(logits[:k], dim=0).tolist()]
        for i in range(k):
            if not reachable[i]:
                logits[i] = -1e4
        if all(not reachable[i] for i in range(k)):   # every reachable object is past the SRE's slots
            reply["reason"] = f"no reachable object among the first {k} slots"
            return -1
        return int(torch.argmax(logits[:k]).item())

    def _sre_inputs(self, color, target, masks, bboxes):
        """Same tensors as Policy.get_unveiler_inputs (the sim evaluation path)."""
        scene = general_utils.resize_mask(color).mean(axis=2)
        scene_t = torch.FloatTensor(scene).unsqueeze(0).to(self.device)
        target_t = torch.FloatTensor(general_utils.resize_mask(target)).unsqueeze(0).to(self.device)

        k = min(len(masks), self.num_patches)
        masks_t = torch.zeros((1, self.num_patches, 1, SRE_SIZE, SRE_SIZE), device=self.device)
        bboxes_t = torch.zeros((1, self.num_patches, 4), device=self.device)
        for i in range(k):
            masks_t[0, i, 0] = torch.FloatTensor(general_utils.resize_mask(masks[i]))
            bboxes_t[0, i] = torch.FloatTensor(general_utils.resize_bbox(bboxes[i]))
        return scene_t, target_t, masks_t, bboxes_t

    def _select_heuristic(self, target, masks, target_index, reachable) -> int:
        n = len(masks)
        if target_index >= 0:
            pool, t_idx = masks, target_index
        else:   # hidden target: rank with its prior mask added as an extra object, then drop it
            pool, t_idx = list(masks) + [target], n
        try:
            order = grasping.find_obstacles_to_remove(t_idx, pool)
        except (ValueError, ZeroDivisionError, IndexError):
            order = []
        order = [i for i in order if i < n]
        # find_obstacles_to_remove returns only [target] for <= 3 objects; rank the rest by distance to the target
        t_c = _centroid(target)
        rest = sorted((i for i in range(n) if i not in order),
                      key=lambda i: np.hypot(*np.subtract(_centroid(masks[i]), t_c)))
        for i in order + rest:
            if reachable[i]:
                return i
        return -1

    def _select_gpt(self, target, masks, reachable) -> int:
        if self._gpt is None:
            from baseline.gpt import GPTRemovalPredictor
            self._gpt = GPTRemovalPredictor()
        prompts = None
        unreachable = [i for i, r in enumerate(reachable) if not r]
        if unreachable:
            prompts = ["an object blocking access to the target", "an obstacle preventing grasping the target",
                       "the first object that should be removed", "an object directly obstructing the target",
                       "the most important obstacle to remove",
                       f"Objects {unreachable} are out of the robot's reach: never choose them."]
        return int(self._gpt.predict(list(masks), target, prompts=prompts))

    def _select_clip(self, target, masks, candidates) -> int:
        if self._clip is None:
            from baseline.clip_eval import ZeroShotCLIPRemovalPredictor
            self._clip = ZeroShotCLIPRemovalPredictor()
        sub = [masks[i].astype(np.float32) / 255.0 for i in candidates]
        return candidates[self._clip.predict_removal(sub, target.astype(np.float32) / 255.0)]

    # ── geometry helpers ─────────────────────────────────────────────────────────────────────────────────────────

    def _roi(self, shape):
        h, w = shape[:2]
        if not self.crop:
            return 0, 0, w, h
        x0, y0, x1, y1 = self.crop
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"crop {self.crop} does not fit a {w}x{h} frame")
        return x0, y0, x1, y1

    @staticmethod
    def _to_sim(mask, roi):
        x0, y0, x1, y1 = roi
        m = cv2.resize(np.asarray(mask)[y0:y1, x0:x1].astype(np.uint8), (SIM_SIZE, SIM_SIZE),
                       interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8) * 255

    @staticmethod
    def _to_frame(mask_sim, shape, roi):
        x0, y0, x1, y1 = roi
        full = np.zeros(shape[:2], np.uint8)
        full[y0:y1, x0:x1] = cv2.resize(mask_sim, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        return full

    @staticmethod
    def _bbox_to_frame(b, roi):
        x0, y0, x1, y1 = roi
        sx, sy = (x1 - x0) / SIM_SIZE, (y1 - y0) / SIM_SIZE
        return [round(b[0] * sx + x0, 1), round(b[1] * sy + y0, 1), round(b[2] * sx + x0, 1), round(b[3] * sy + y0, 1)]

    def _log_step(self, step, method, image_bgr, target_mask, overlay, masks, reply) -> str:
        name = f"step_{int(step):02d}_{method}" if step is not None else f"step_{time.strftime('%H%M%S')}_{method}"
        d = self.log_dir / name
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "frame.jpg"), image_bgr)
        cv2.imwrite(str(d / "target_mask.png"), (np.asarray(target_mask) > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(d / "overlay.jpg"), overlay)
        np.savez_compressed(d / "masks_sim.npz", masks=np.array(masks, dtype=np.uint8).reshape(-1, SIM_SIZE, SIM_SIZE))
        with open(d / "reply.json", "w") as f:
            json.dump({k: v for k, v in reply.items() if k not in ("chosen_mask", "overlay")}, f, indent=2)
        return str(d)


def _centroid(mask) -> List[int]:
    m = cv2.moments((np.asarray(mask) > 0).astype(np.uint8))
    if m["m00"] == 0:
        return [0, 0]
    return [int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])]


def _match_target(target, masks) -> int:
    """Index of the segmented object that is the target (best IoU above TARGET_MATCH_IOU), else -1."""
    t = target > 0
    best, best_iou = -1, TARGET_MATCH_IOU
    for i, m in enumerate(masks):
        o = m > 0
        union = np.logical_or(t, o).sum()
        iou = np.logical_and(t, o).sum() / union if union else 0.0
        if iou >= best_iou:
            best, best_iou = i, iou
    return best


def draw_overlay(image_bgr, masks, target_mask, chosen, target_index, reachable) -> np.ndarray:
    """Frame with every object outlined and numbered: target red, chosen object green, unreachable grey."""
    out = image_bgr.copy()
    for i, m in enumerate(masks):
        color = (160, 160, 160) if not reachable[i] else (0, 215, 255)
        contours, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)
        cx, cy = _centroid(m)
        cv2.putText(out, str(i), (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(out, str(i), (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    contours, _ = cv2.findContours((np.asarray(target_mask) > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    if chosen >= 0:
        contours, _ = cv2.findContours((masks[chosen] > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 200, 0), 3)
    label = f"chosen={chosen} target={'hidden' if target_index < 0 else target_index}"
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out
