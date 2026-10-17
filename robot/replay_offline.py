#!/usr/bin/env python3
"""Replay saved real frames through every selector (and several SRE-RL checkpoints) in one pass.

The episodes captured with ``unveiler_session.py --executor none`` each ran one method. This script takes each
episode's first frame + target mask (or every step with --all-steps), segments it once per method exactly as the
server does, and records every method's choice on the same objects. Two outputs matter:

    replay.csv      one row per (scene, method): chosen index, is_target, top probability, top-1 minus top-2 margin
    scenes/*.jpg    each scene with every object numbered (target red) and a panel listing each method's choice
    labels.csv      one row per scene with an empty `correct` column for you to fill (only created if missing)

Label a scene by writing the indices a sensible operator would remove next, separated by ';' (e.g. ``2`` or ``2;3``),
or ``t`` when the target itself should be grasped now. Then score:

    python -m robot.replay_offline --session-dir save/real_eval/offline_check \
        --warp 88 71 501 38 639 479 54 479 \
        --rl-ckpts save/sre_rl/sre_rl_1000.pt save/sre_rl/sre_rl_3000.pt save/sre_rl/sre_rl_6000.pt
    # fill labels.csv, then:
    python -m robot.replay_offline --session-dir save/real_eval/offline_check --score

Use the same --warp/--crop the server had when the frames were captured. The script compares its rectified view with
the saved sim_view.jpg and warns when they differ, which means the corners do not match.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("replay_offline")
BASE_METHODS = ["sre", "sre_il", "heuristic", "random"]


def find_scenes(session_dir: Path, all_steps: bool):
    scenes = []
    for ep in sorted(session_dir.glob("episode_*")):
        steps = sorted(d for d in ep.glob("step_*") if (d / "frame.jpg").exists() and (d / "target_mask.png").exists())
        if not steps:
            continue
        for st in (steps if all_steps else steps[:1]):
            reply = json.load(open(st / "reply.json")) if (st / "reply.json").exists() else {}
            scenes.append({"episode": ep.name, "step": st.name.split("_")[1], "dir": st,
                           "target_visible": reply.get("target_visible", True)})
    return scenes


def scene_panel(frame, masks, target_index, choices, title):
    """Frame with every object outlined + numbered, and a side panel of method -> choice."""
    from robot.backend import _centroid
    out = frame.copy()
    for i, m in enumerate(masks):
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 0, 255) if i == target_index else (0, 215, 255), 3 if i == target_index else 1)
        cx, cy = _centroid(m)
        for col, th in (((255, 255, 255), 4), ((0, 0, 0), 2)):
            cv2.putText(out, str(i), (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th)
    panel = np.full((out.shape[0], 300, 3), 255, np.uint8)
    lines = [title, f"target = {target_index if target_index >= 0 else 'hidden'} (red)", ""]
    lines += [f"{m:<16} -> {'t' if c.get('is_target') else c['chosen']}"
              + (f"  p={c['top_prob']:.2f}" if c.get("top_prob") is not None else "") for m, c in choices.items()]
    for k, line in enumerate(lines):
        cv2.putText(panel, line, (8, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return np.hstack([out, panel])


def replay(args):
    import torch
    from robot.backend import UnveilerBackend
    from policy.sre_actor_critic import SREActorCritic

    session = Path(args.session_dir)
    out_dir = Path(args.out or session / "replay")
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    scenes = find_scenes(session, args.all_steps)
    if not scenes:
        raise SystemExit(f"no episode_*/step_*/frame.jpg under {session}")
    logger.info("%d scenes from %s", len(scenes), session)

    backend = UnveilerBackend(device=args.device, sre_rl_ckpt=args.sre_rl_ckpt, sre_il_ckpt=args.sre_il_ckpt,
                              seg_threshold=args.seg_threshold, crop=args.crop, warp=args.warp,
                              output_dir=str(out_dir / "server_logs"), seed=args.seed)
    backend.target_min_prob = args.target_min_prob
    default_rl = backend.sre_rl
    rl_models = {}
    for path in args.rl_ckpts or []:
        m = SREActorCritic(_model_args(backend)).to(backend.device)
        ck = torch.load(path, map_location=backend.device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
        m.eval()
        rl_models[f"sre@{Path(path).stem.replace('sre_rl_', '')}"] = m
    default_il = backend.sre_il
    il_models = {}
    for path in args.il_ckpts or []:          # extra SpatialEncoder checkpoints (IL or expert iteration)
        from robot.backend import SpatialEncoder
        m = SpatialEncoder(_model_args(backend)).to(backend.device)
        m.load_state_dict(torch.load(path, map_location=backend.device))
        m.eval()
        il_models[f"il@{Path(path).parent.name}/{Path(path).stem}"] = m

    methods = list(args.methods) + list(rl_models) + list(il_models)
    rows = []
    for k, sc in enumerate(scenes):
        frame = cv2.imread(str(sc["dir"] / "frame.jpg"))
        target = cv2.imread(str(sc["dir"] / "target_mask.png"), cv2.IMREAD_GRAYSCALE)
        backend.reset(sc["episode"], k)
        choices, masks_ref, t_ref = {}, None, -1
        for method in methods:
            backend.sre_rl = rl_models.get(method, default_rl)
            backend.sre_il = il_models.get(method, default_il)
            base = "sre" if method.startswith("sre@") else "sre_il" if method.startswith("il@") else method
            r = backend.select(frame, target, method=base,
                               target_visible=sc["target_visible"], step=int(sc["step"]))
            probs = r.get("probs")
            srt = sorted(probs, reverse=True) if probs else None
            row = {"episode": sc["episode"], "step": sc["step"], "method": method, "n_objects": r["num_objects"],
                   "target_index": r["target_index"], "chosen": r["chosen_index"], "is_target": r["is_target"],
                   "top_prob": srt[0] if srt else None,
                   "margin": (srt[0] - srt[1]) if srt and len(srt) > 1 else None,
                   "probs": json.dumps(probs) if probs else "", "select_ms": r["timing_ms"]["select"]}
            rows.append(row)
            choices[method] = row
            if masks_ref is None:
                masks_ref = np.load(Path(r["log_dir"]) / "masks_sim.npz")["masks"]
                t_ref = r["target_index"]
                _check_warp(sc["dir"], Path(r["log_dir"]), sc["episode"])
        backend.sre_rl, backend.sre_il = default_rl, default_il
        # full-frame masks for the panel, from the first method's (identical) segmentation
        H, H_inv = backend._homography(frame.shape)
        full = [backend._to_frame(m, frame.shape, H_inv) for m in masks_ref]
        title = f"{sc['episode']} step {sc['step']}"
        cv2.imwrite(str(out_dir / "scenes" / f"{sc['episode']}_s{sc['step']}.jpg"),
                    scene_panel(frame, full, t_ref, choices, title))
        logger.info("%s: %s", title, {m: ("t" if c["is_target"] else c["chosen"]) for m, c in choices.items()})

    fields = list(rows[0].keys())
    with open(out_dir / "replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    labels = out_dir / "labels.csv"
    if not labels.exists():
        with open(labels, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "step", "n_objects", "target_index", "correct", "note"])
            seen = set()
            for r in rows:
                key = (r["episode"], r["step"])
                if key not in seen:
                    seen.add(key)
                    w.writerow([r["episode"], r["step"], r["n_objects"], r["target_index"], "", ""])
    logger.info("wrote %s, %s and %s/scenes/", out_dir / "replay.csv", labels, out_dir)
    print_unlabelled_stats(rows, methods)


def _model_args(backend):
    from argparse import Namespace
    return Namespace(device=backend.device, num_patches=backend.num_patches, sequence_length=1)


def _check_warp(capture_dir: Path, replay_dir: Path, name):
    a, b = cv2.imread(str(capture_dir / "sim_view.jpg")), cv2.imread(str(replay_dir / "sim_view.jpg"))
    if a is None or b is None:
        return
    diff = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    if diff > 12:
        logger.warning("%s: rectified view differs from the capture (mean abs diff %.1f): is --warp/--crop the "
                       "one the server used?", name, diff)


def print_unlabelled_stats(rows, methods):
    """What can be said without labels: confidence, and agreement with the heuristic and with each other."""
    by = {m: [r for r in rows if r["method"] == m] for m in methods}
    heur = {(r["episode"], r["step"]): r for r in by.get("heuristic", [])}
    print(f"\n{'method':<14} {'scenes':>6} {'mean top p':>10} {'mean margin':>11} {'= heuristic':>11} {'target now':>10}")
    for m in methods:
        rs = by[m]
        tp = [r["top_prob"] for r in rs if r["top_prob"] is not None]
        mg = [r["margin"] for r in rs if r["margin"] is not None]
        agree = [r["chosen"] == heur[(r["episode"], r["step"])]["chosen"] for r in rs if (r["episode"], r["step"]) in heur]
        print(f"{m:<14} {len(rs):>6} {np.mean(tp) if tp else float('nan'):>10.2f} "
              f"{np.mean(mg) if mg else float('nan'):>11.2f} {np.mean(agree) if agree else float('nan'):>11.0%} "
              f"{np.mean([r['is_target'] in (True, 'True') for r in rs]):>10.0%}")
    print("\nFill labels.csv (`correct` = indices to remove next, ';'-separated, or t), then run with --score.")


def score(args):
    out_dir = Path(args.out or Path(args.session_dir) / "replay")
    rows = list(csv.DictReader(open(out_dir / "replay.csv")))
    labels = {(l["episode"], l["step"]): l for l in csv.DictReader(open(out_dir / "labels.csv"))}
    labelled = {k: v for k, v in labels.items() if v["correct"].strip()}
    if not labelled:
        raise SystemExit(f"no labels in {out_dir / 'labels.csv'} yet")
    methods = list(dict.fromkeys(r["method"] for r in rows))
    # baseline that needs no model: grasp the target now (the bar every method must clear)
    first = [r for r in rows if r["method"] == methods[0]]
    rows = rows + [{**r, "method": "always_target", "is_target": "True", "chosen": r["target_index"]} for r in first]
    methods.append("always_target")
    print(f"{len(labelled)} labelled scenes\n")
    print(f"{'method':<14} {'correct':>9} {'acc':>6}   by density (n objects)")
    rng = np.random.RandomState(0)
    for m in methods:
        hits, dens = [], {}
        for r in rows:
            key = (r["episode"], r["step"])
            if r["method"] != m or key not in labelled:
                continue
            ok_set = {s.strip() for s in labelled[key]["correct"].split(";") if s.strip()}
            pick = "t" if r["is_target"] == "True" else r["chosen"]
            hit = pick in ok_set
            hits.append(hit)
            n = int(r["n_objects"])
            b = "2-4" if n <= 4 else "5-8" if n <= 8 else "9+"
            dens.setdefault(b, []).append(hit)
        if not hits:
            continue
        boot = [np.mean(rng.choice(hits, len(hits))) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        per = "  ".join(f"{b}: {sum(v)}/{len(v)}" for b, v in sorted(dens.items()))
        print(f"{m:<14} {sum(hits):>4}/{len(hits):<4} {np.mean(hits):>6.0%}   [{lo:.0%}, {hi:.0%}]  {per}")
    print("\n[..] = 95% bootstrap interval. With ~20 scenes, differences under ~20 points are within noise.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <session-dir>/replay")
    ap.add_argument("--score", action="store_true", help="score replay.csv against the filled labels.csv")
    ap.add_argument("--methods", nargs="+", default=BASE_METHODS,
                    help="server methods: sre sre_il heuristic random clip gpt4o")
    ap.add_argument("--rl-ckpts", nargs="*", default=[], help="extra SRE-RL checkpoints, reported as sre@<episode>")
    ap.add_argument("--il-ckpts", nargs="*", default=[],
                    help="extra SRE (SpatialEncoder) checkpoints, e.g. expert-iteration ones, reported as il@<dir>/<name>")
    ap.add_argument("--all-steps", action="store_true", help="replay every saved step, not just step 0")
    ap.add_argument("--crop", type=int, nargs=4, default=None)
    ap.add_argument("--warp", type=float, nargs=8, default=None)
    ap.add_argument("--seg-threshold", type=float, default=0.97)
    ap.add_argument("--target-min-prob", type=float, default=None,
                    help="SRE methods grasp the target only at or above this probability (backend.target_min_prob)")
    ap.add_argument("--sre-rl-ckpt", default="save/sre_rl/sre_rl_best.pt")
    ap.add_argument("--sre-il-ckpt", default="save/sre/sre_model_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.score:
        score(args)
    else:
        replay(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
