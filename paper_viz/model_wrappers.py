"""
Thin wrappers around SRE, CLIP, GPT-4o, and the heuristic so that
evaluate_scenes.py has a uniform call signature for every predictor:

    obstacle_id: int = predictor.predict(scene_data)

where scene_data is a dict returned by SceneData.load().
"""

import argparse

import cv2
import numpy as np
import torch

from baseline.clip_eval import ZeroShotCLIPRemovalPredictor
from baseline.gpt import GPTRemovalPredictor
import policy.grasping as grasping
import utils.general_utils as general_utils


# ---------------------------------------------------------------------------
# Shared SceneData helper
# ---------------------------------------------------------------------------

class SceneData:
    """Holds everything loaded for one segmented scene."""

    def __init__(
        self,
        scene_id: str,
        color_image: np.ndarray,
        scene_mask: np.ndarray,
        target_mask: np.ndarray,
        object_masks: list,
        bboxes: list,
        target_id: int,
    ):
        self.scene_id = scene_id
        self.color_image = color_image
        self.scene_mask = scene_mask
        self.target_mask = target_mask
        self.object_masks = object_masks
        self.bboxes = bboxes
        self.target_id = target_id

    def object_crop(self, mask: np.ndarray) -> np.ndarray:
        """Extract a cropped/masked view of an object against the scene."""
        return general_utils.extract_target_crop2(mask, self.color_image)


# ---------------------------------------------------------------------------
# Heuristic predictor
# ---------------------------------------------------------------------------

class HeuristicPredictor:
    """Geometric heuristic: periphery + proximity to target."""

    def predict(self, scene: SceneData) -> int:
        order = grasping.find_obstacles_to_remove(scene.target_id, scene.object_masks)
        # find_obstacles_to_remove may put target_id first in degenerate cases;
        # return the top-ranked candidate regardless — evaluation handles context.
        return int(order[0])


# ---------------------------------------------------------------------------
# SRE predictor
# ---------------------------------------------------------------------------

class SREPredictor:
    """
    Wraps SREActorCritic for standalone inference.
    Replicates the preprocessing from Policy.get_unveiler_inputs and uses
    SREActorCritic.act (deterministic=True) matching exploit_unveiler_rl.
    """

    def __init__(
        self,
        model_path: str = "save/sre_rl/sre_rl_best.pt",
        num_patches: int = 10,
        device: torch.device = None,
    ):
        from policy.sre_actor_critic import SREActorCritic

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_patches = num_patches

        args = argparse.Namespace(device=self.device, num_patches=num_patches)
        # Construct without pretrained path — weights loaded manually below
        self.model = SREActorCritic(args).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    @torch.no_grad()
    def predict(self, scene: SceneData) -> int:
        # --- Scene image (grayscale) ---
        scene_img = general_utils.resize_mask(scene.color_image).mean(axis=2)
        proc_scene = torch.FloatTensor(scene_img).unsqueeze(0).to(self.device)

        # --- Target mask ---
        proc_target = torch.FloatTensor(
            general_utils.resize_mask(scene.target_mask)
        ).unsqueeze(0).to(self.device)

        # --- Per-object masks + bboxes ---
        proc_masks = []
        resized_bboxes = []
        for mask, bbox in zip(scene.object_masks, scene.bboxes):
            proc_masks.append(
                torch.FloatTensor(general_utils.resize_mask(mask)).unsqueeze(0)
            )
            resized_bboxes.append(general_utils.resize_bbox(bbox))

        proc_masks = torch.stack(proc_masks).to(self.device)
        bboxes_t = torch.FloatTensor(resized_bboxes).to(self.device)

        # --- Pad / truncate to num_patches ---
        n = proc_masks.shape[0]
        if n < self.num_patches:
            padding = self.num_patches - n
            proc_masks = proc_masks.unsqueeze(0)  # (1, n, C, H, W)
            proc_masks = torch.nn.functional.pad(
                proc_masks, (0, 0, 0, 0, 0, 0, 0, padding, 0, 0)
            )
            bboxes_t = bboxes_t.unsqueeze(0)
            bboxes_t = torch.nn.functional.pad(bboxes_t, (0, 0, 0, padding))
        else:
            proc_masks = proc_masks[: self.num_patches].unsqueeze(0)
            bboxes_t = bboxes_t[: self.num_patches].unsqueeze(0)

        action, _, _ = self.model.act(proc_scene, proc_target, proc_masks, bboxes_t, deterministic=True)
        return int(action.item())


# ---------------------------------------------------------------------------
# CLIP predictor  (thin pass-through — underlying class already clean)
# ---------------------------------------------------------------------------

class CLIPPredictor:
    def __init__(self, clip_model: str = "ViT-B/32"):
        self._predictor = ZeroShotCLIPRemovalPredictor(clip_model=clip_model)

    def predict(self, scene: SceneData) -> int:
        # Normalise masks to 0-1 float for CLIP (the underlying impl expects this)
        float_masks = [m.astype(np.float32) / 255.0 for m in scene.object_masks]
        float_target = scene.target_mask.astype(np.float32) / 255.0
        return int(self._predictor.predict_removal(float_masks, float_target))


# ---------------------------------------------------------------------------
# GPT-4o predictor  (improved prompt; unchanged structured-output contract)
# ---------------------------------------------------------------------------

class GPT4oPredictor:
    """
    Sends per-object crops + target crop to GPT-4o with vision.
    Uses a tighter, robotics-specific system prompt.
    """

    def __init__(self):
        self._predictor = GPTRemovalPredictor()
        # Override the system prompt for a clearer task description
        self._predictor.predict = self._predict_patched.__get__(
            self._predictor, GPTRemovalPredictor
        )

    def _predict_patched(self_inner, objects, target, prompts=None):
        """
        Patched version of GPTRemovalPredictor.predict with an improved prompt.
        'self_inner' refers to the GPTRemovalPredictor instance.
        """
        import json
        object_data_uris = [self_inner.np_to_data_uri(obj) for obj in objects]
        target_data_uri = self_inner.np_to_data_uri(target)

        object_contents = []
        for i, uri in enumerate(object_data_uris):
            object_contents.append({"type": "text", "text": f"Object {i}:"})
            object_contents.append({"type": "image_url", "image_url": {"url": uri}})

        response = self_inner.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are assisting a robot arm in a tabletop manipulation task. "
                        "The robot must grasp a target object. "
                        "Objects are scattered on a table and may be blocking access to the target. "
                        "Your task: given images of individual objects and the target, "
                        "identify which single object is the most important obstacle to remove first "
                        "so the robot can reach the target. "
                        "Rules:\n"
                        "- Return ONLY the integer index of the chosen obstacle.\n"
                        "- Do NOT choose the target object itself as the obstacle.\n"
                        "- Choose the object that most directly blocks or is closest to the target.\n"
                        "- Output ONLY JSON with field `chosen_index`."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Objects in the scene (indexed from 0):"},
                        *object_contents,
                        {"type": "text", "text": "Target object to grasp:"},
                        {"type": "image_url", "image_url": {"url": target_data_uri}},
                        {
                            "type": "text",
                            "text": (
                                "Which object (by index) should the robot remove first "
                                "to clear a path to the target? "
                                "Return JSON: {\"chosen_index\": <int>}"
                            ),
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "best_obstacle",
                    "schema": {
                        "type": "object",
                        "properties": {"chosen_index": {"type": "integer"}},
                        "required": ["chosen_index"],
                    },
                },
            },
            temperature=0,
        )

        chosen_index = json.loads(response.choices[0].message.content)["chosen_index"]
        print("GPT-4o response:", chosen_index)
        return chosen_index

    def predict(self, scene: SceneData) -> int:
        object_crops = [scene.object_crop(m) for m in scene.object_masks]
        target_crop = scene.object_crop(scene.target_mask)
        # Convert BGR→RGB for display models
        object_crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in object_crops]
        target_crop_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
        return int(self._predictor.predict(object_crops_rgb, target_crop_rgb))
