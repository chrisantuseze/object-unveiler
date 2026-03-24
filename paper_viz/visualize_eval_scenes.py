"""
Lab-side visualisation script.

For each evaluated scene produces:
  • A 2×2 figure:  Scene | Target | Pred Obstacle (<model>) | Heur Obstacle
  • Saved to <scenes_dir>/figs/<scene_id>_<model>.png

Usage:
    # View one model interactively (show each figure, press any key to advance)
    python -m paper_viz.visualize_eval_scenes \
        --scenes_dir save/real-eval-scenes \
        --model sre

    # Save all figures for all models without displaying them
    python -m paper_viz.visualize_eval_scenes \
        --scenes_dir save/real-eval-scenes \
        --model all \
        --no_show

    # Only render a specific scene
    python -m paper_viz.visualize_eval_scenes \
        --scenes_dir save/real-eval-scenes \
        --model clip \
        --scene transition_000003
"""

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from paper_viz.model_wrappers import SceneData
from trainer.memory import ReplayBuffer
import utils.general_utils as general_utils


SCENES_DIR_DEFAULT = "paper_viz/real-eval-scenes-2-4-"
MODELS = ["sre", "clip", "gpt", "heuristic"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(scenes_dir: str) -> dict:
    path = os.path.join(scenes_dir, "results.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No results file found at '{path}'. "
            "Run evaluate_scenes.py first."
        )
    return pickle.load(open(path, "rb"))


def get_segmented_dirs(scenes_dir: str) -> list[str]:
    entries = sorted(os.listdir(scenes_dir))
    return [
        name
        for name in entries
        if name.startswith("transition_")
        and os.path.isdir(os.path.join(scenes_dir, name))
        and os.path.exists(os.path.join(scenes_dir, name, "num_objects"))
    ]


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


def make_crop(mask: np.ndarray, color_image: np.ndarray) -> np.ndarray:
    crop = general_utils.extract_target_crop2(mask, color_image)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def make_figure(
    scene: SceneData,
    pred_mask: np.ndarray,
    pred_label: str,
    heur_mask: np.ndarray,
) -> plt.Figure:
    color_rgb = cv2.cvtColor(scene.color_image, cv2.COLOR_BGR2RGB)
    target_crop = make_crop(scene.target_mask, scene.color_image)
    pred_crop = make_crop(pred_mask, scene.color_image)
    heur_crop = make_crop(heur_mask, scene.color_image)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(scene.scene_id, fontsize=11)

    axes[0][0].imshow(color_rgb)
    axes[0][0].set_title("Scene - Color")
    axes[0][0].axis("off")

    axes[0][1].imshow(target_crop)
    axes[0][1].set_title(f"Target (id={scene.target_id})")
    axes[0][1].axis("off")

    axes[1][0].imshow(pred_crop)
    axes[1][0].set_title(f"Pred Obstacle [{pred_label.upper()}]")
    axes[1][0].axis("off")

    axes[1][1].imshow(heur_crop)
    axes[1][1].set_title("Heur Obstacle")
    axes[1][1].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def visualize(args):
    results = load_results(args.scenes_dir)
    dir_ids = get_segmented_dirs(args.scenes_dir)

    if args.scene:
        dir_ids = [d for d in dir_ids if d == args.scene]
        if not dir_ids:
            print(f"Scene '{args.scene}' not found or not segmented.")
            return

    models_to_run = MODELS if args.model == "all" else [args.model]

    figs_dir = os.path.join(args.scenes_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    memory = ReplayBuffer(args.scenes_dir)

    for idx, scene_id in enumerate(dir_ids):
        if scene_id not in results:
            print(f"[{idx + 1}] {scene_id} — not in results, skipping.")
            continue

        entry = results[scene_id]

        try:
            scene = load_scene(memory, dir_ids, idx)
        except Exception as e:
            print(f"[{idx + 1}] {scene_id} — load error: {e}")
            continue

        heur_id = entry.get("heuristic")
        if heur_id is None or heur_id >= len(scene.object_masks):
            print(f"[{idx + 1}] {scene_id} — invalid heuristic id, skipping.")
            continue
        heur_mask = scene.object_masks[heur_id]

        print(f"[{idx + 1}/{len(dir_ids)}] {scene_id}")

        for model in models_to_run:
            pred_id = entry.get(model)
            if pred_id is None:
                print(f"  [{model.upper()}] no prediction available, skipping.")
                continue
            if pred_id >= len(scene.object_masks):
                print(f"  [{model.upper()}] pred_id={pred_id} out of range, skipping.")
                continue

            pred_mask = scene.object_masks[pred_id]

            print(
                f"  [{model.upper()}] pred={pred_id}  heur={heur_id}  "
                f"target={scene.target_id}  n_objects={len(scene.object_masks)}"
            )

            fig = make_figure(scene, pred_mask, model, heur_mask)

            save_path = os.path.join(figs_dir, f"{scene_id}_{model}.png")
            fig.savefig(save_path, dpi=120, bbox_inches="tight")

            if not args.no_show:
                plt.show()
            plt.close(fig)

    print(f"Figures saved to '{figs_dir}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise real-scene evaluation results."
    )
    parser.add_argument(
        "--scenes_dir",
        type=str,
        default=SCENES_DIR_DEFAULT,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sre",
        choices=MODELS + ["all"],
        help="Which model's prediction to show in the Pred Obstacle panel.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="",
        help="Only visualise a specific scene (e.g. transition_000003).",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Save figures without calling plt.show() (useful on headless machines).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
