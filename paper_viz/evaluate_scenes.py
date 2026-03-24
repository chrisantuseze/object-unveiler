"""
Lab-side evaluation runner.

Loads every segmented scene from the shared scenes directory, runs all four
predictors (SRE, CLIP, GPT-4o, Heuristic), and saves a results file that the
visualisation script can consume.

Usage:
    python -m paper_viz.evaluate_scenes \
        --scenes_dir save/real-eval-scenes \
        --sre_model  save/sre/sre_model_best.pt \
        --skip_gpt          # optional: skip GPT-4o to save API calls
        --limit 20          # optional: only process first N scenes

Results are saved to:
    <scenes_dir>/results.pkl   – dict keyed by scene_id
"""

from __future__ import annotations


import argparse
import os
import pickle

import cv2

from paper_viz.model_wrappers import (
    CLIPPredictor,
    GPT4oPredictor,
    HeuristicPredictor,
    SceneData,
    SREPredictor,
)
from trainer.memory import ReplayBuffer


SCENES_DIR_DEFAULT = "save/real-eval-scenes"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_segmented_dirs(scenes_dir: str) -> list[str]:
    """Return sorted transition folder *names* that are fully segmented."""
    entries = sorted(os.listdir(scenes_dir))
    ready = []
    for name in entries:
        if not name.startswith("transition_"):
            continue
        folder = os.path.join(scenes_dir, name)
        if os.path.isdir(folder) and os.path.exists(
            os.path.join(folder, "num_objects")
        ):
            ready.append(name)
    return ready


def load_scene(memory: ReplayBuffer, dir_ids: list[str], idx: int) -> SceneData:
    color_image, scene_mask, target_mask, bboxes, target_id, object_masks = (
        memory.load_seg_data(dir_ids, idx)
    )
    color_image = cv2.resize(color_image, (400, 400))
    return SceneData(
        scene_id=dir_ids[idx],
        color_image=color_image,
        scene_mask=scene_mask,
        target_mask=target_mask,
        object_masks=object_masks,
        bboxes=bboxes,
        target_id=target_id,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_evaluation(args):
    dir_ids = get_segmented_dirs(args.scenes_dir)

    if not dir_ids:
        print("No segmented scenes found. Run segment_scenes.py first.")
        return

    if args.limit:
        dir_ids = dir_ids[: args.limit]

    print(f"Evaluating {len(dir_ids)} scene(s) from '{args.scenes_dir}'.")

    memory = ReplayBuffer(args.scenes_dir)

    # Load existing results so we can resume a partial run
    results_path = os.path.join(args.scenes_dir, "results.pkl")
    if os.path.exists(results_path):
        results = pickle.load(open(results_path, "rb"))
        print(f"  Resuming: {len(results)} scene(s) already in results file.")
    else:
        results = {}

    # Initialise predictors
    heuristic = HeuristicPredictor()
    sre = SREPredictor(model_path=args.sre_model)
    clip = CLIPPredictor()
    gpt = None if args.skip_gpt else GPT4oPredictor()

    for idx, scene_id in enumerate(dir_ids):
        if scene_id in results:
            print(f"[{idx + 1}/{len(dir_ids)}] {scene_id} — already evaluated, skipping.")
            continue

        print(f"[{idx + 1}/{len(dir_ids)}] {scene_id} ...", end=" ", flush=True)

        try:
            scene = load_scene(memory, dir_ids, idx)
        except Exception as e:
            print(f"[ERROR loading] {e}")
            continue

        entry = {
            "scene_id": scene_id,
            "target_id": scene.target_id,
            "num_objects": len(scene.object_masks),
        }

        try:
            entry["heuristic"] = heuristic.predict(scene)
        except Exception as e:
            print(f"\n  [WARN] heuristic failed: {e}")
            entry["heuristic"] = None

        try:
            entry["sre"] = sre.predict(scene)
        except Exception as e:
            print(f"\n  [WARN] SRE failed: {e}")
            entry["sre"] = None

        try:
            entry["clip"] = clip.predict(scene)
        except Exception as e:
            print(f"\n  [WARN] CLIP failed: {e}")
            entry["clip"] = None

        if gpt is not None:
            try:
                entry["gpt"] = gpt.predict(scene)
            except Exception as e:
                print(f"\n  [WARN] GPT-4o failed: {e}")
                entry["gpt"] = None
        else:
            entry["gpt"] = None

        results[scene_id] = entry
        # Save after every scene so progress is never lost
        pickle.dump(results, open(results_path, "wb"))

        print(
            f"heuristic={entry['heuristic']}  "
            f"sre={entry['sre']}  "
            f"clip={entry['clip']}  "
            f"gpt={entry['gpt']}"
        )

    print(f"\nFinished. Results saved to '{results_path}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run model evaluations on real scenes.")
    parser.add_argument(
        "--scenes_dir",
        type=str,
        default=SCENES_DIR_DEFAULT,
    )
    parser.add_argument(
        "--sre_model",
        type=str,
        default="save/sre_rl/sre_rl_best.pt",
    )
    parser.add_argument(
        "--skip_gpt",
        action="store_true",
        help="Skip GPT-4o (saves API calls during debugging).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N scenes (0 = all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
