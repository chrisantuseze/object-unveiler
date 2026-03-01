"""
Lab-side segmentation & preprocessing pipeline.

Iterates over every transition_* folder in the scenes directory that has a
scene_image.png but has NOT yet been segmented (no 'num_objects' pickle).

For each scene it:
  1. Runs Mask R-CNN (real-robot threshold) to get per-object masks & bboxes.
  2. Randomly selects a target object.
  3. Writes all outputs in the exact format that ReplayBuffer.load_seg_data
     expects, so the evaluation pipeline needs no special-case code.

Usage:
    python -m paper_viz.segment_scenes \
        --scenes_dir save/real-eval-scenes \
        --seed 42
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import tempfile

import cv2
import numpy as np

from mask_rg.object_segmenter import ObjectSegmenter


SCENES_DIR_DEFAULT = "save/real-eval-scenes"


def get_unsegmented_dirs(scenes_dir: str) -> list[str]:
    """Return sorted list of transition folder *names* that still need segmenting."""
    entries = sorted(os.listdir(scenes_dir))
    pending = []
    for name in entries:
        if not name.startswith("transition_"):
            continue
        folder = os.path.join(scenes_dir, name)
        if not os.path.isdir(folder):
            continue
        color_path = os.path.join(folder, "scene_image.png")
        done_path = os.path.join(folder, "num_objects")
        if os.path.exists(color_path) and not os.path.exists(done_path):
            pending.append(name)
    return pending


def segment_scene(
    folder: str,
    segmenter: ObjectSegmenter,
    rng: random.Random,
) -> bool:
    """
    Segment a single scene folder. Returns True on success, False if skipped.
    """
    color_path = os.path.join(folder, "scene_image.png")
    color_image = cv2.imread(color_path)
    if color_image is None:
        print(f"  [WARN] Could not read {color_path}, skipping.")
        return False

    # Resize to the expected 400×400 used throughout the project
    color_image = cv2.resize(color_image, (400, 400))

    # Run segmentation; direct mask writes from from_maskrcnn go to a temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            object_masks, scene_mask, _, bboxes = segmenter.from_maskrcnn(
                color_image, dir=tmpdir, bbox=True
            )
        except Exception as e:
            print(f"  [ERROR] Segmentation failed: {e}")
            return False

    if len(object_masks) == 0:
        print(f"  [WARN] No objects detected, skipping.")
        return False

    # Random target selection
    target_id = rng.randint(0, len(object_masks) - 1)
    target_mask = object_masks[target_id]

    # --- Write in ReplayBuffer.store_seg_data format ---
    cv2.imwrite(color_path, color_image)  # overwrite with resized version
    cv2.imwrite(os.path.join(folder, "scene_mask.png"), scene_mask)
    cv2.imwrite(os.path.join(folder, "target_mask.png"), target_mask)

    for i, mask in enumerate(object_masks):
        cv2.imwrite(os.path.join(folder, f"object_{i}.png"), mask)

    pickle.dump(len(object_masks), open(os.path.join(folder, "num_objects"), "wb"))
    pickle.dump(target_id, open(os.path.join(folder, "target_id"), "wb"))
    pickle.dump(bboxes, open(os.path.join(folder, "bboxes"), "wb"))

    print(
        f"  Objects: {len(object_masks)}, target_id: {target_id}, "
        f"bboxes: {len(bboxes)}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Segment collected real-robot scenes.")
    parser.add_argument(
        "--scenes_dir",
        type=str,
        default=SCENES_DIR_DEFAULT,
        help="Root directory containing transition_* folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible target selection.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pending = get_unsegmented_dirs(args.scenes_dir)

    if not pending:
        print("No unsegmented scenes found.")
        return

    print(f"Found {len(pending)} scene(s) to segment.")
    segmenter = ObjectSegmenter(is_real=True)

    successes = 0
    for name in pending:
        folder = os.path.join(args.scenes_dir, name)
        print(f"Processing {name} ...")
        if segment_scene(folder, segmenter, rng):
            successes += 1

    print(f"\nDone. {successes}/{len(pending)} scene(s) segmented successfully.")


if __name__ == "__main__":
    main()
