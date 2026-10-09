"""Selection-only evaluation in the DOFBOT twin, through the real robot's server code.

Each held-out twin scene (twin/unveil.py) is rendered from the arm camera and handed to robot/backend.py exactly as
the Jetson would: a camera frame and a target mask. The method answers with a mask; the block behind it is lifted
out and the pile settles (ideal removal, so every failure is the selector's). The episode succeeds when the method
picks the target while the search's top-grasp test passes; a grasp at a still-blocked target ends it as a failure,
as it would on the robot.

    python -m twin.eval --out save/twin_eval/run --n_scenes 200 \
        --il-ckpts save/sre_twin2/sre_exit_it0.pt save/sre_exit_top/sre_exit_it0.pt
    python -m twin.eval --summarize save/twin_eval/run
    python -m twin.eval --layouts twin/real_eval_layouts/layouts.json --out save/twin_eval/real_layouts ...
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from twin.unveil import WARP, UnveilTwin, parse_args as unveil_args   # noqa: E402

MIN_TARGET_PX = 150        # as the Jetson runner: a smaller visible target counts as hidden


def frame_and_ids(ut, rng, aug):
    from twin import augment
    tw = ut.tw
    cam = tw._jitter(tw.episode_camera, tw.cfg["camera"]["jitter_frame"], rng)
    ids = ut.block_ids(cam)
    return cv2.cvtColor(augment.apply(tw.render(cam=cam), aug, rng), cv2.COLOR_RGB2BGR), ids


def run_episode(ut, backend, models, method, scene_state, target, ref_mask, aug, seed, max_steps):
    """One method on one scene. Returns the episode record."""
    from robot.protocol import decode_mask
    rng = np.random.default_rng(seed)                 # same camera jitter and noise for every method
    ut.restore(scene_state)
    t_i = ut.colors.index(target)
    rec = {"method": method, "steps": [], "success": False, "outcome": "step_limit"}
    for step in range(max_steps):
        frame, ids = frame_and_ids(ut, rng, aug)
        visible = (ids == t_i)
        seen = int(visible.sum()) >= MIN_TARGET_PX
        if seen:
            ref_mask = visible.astype(np.uint8) * 255
        ut.start_search(target)
        v0 = ut.value(())
        graspable = ut._state(())[1]
        cands = [c for c in ut.present() if c != target]
        cost = {c: 1 + ut.value((c,)) for c in cands}
        ut.restore(ut.root)
        if step == 0:
            rec["optimal_steps"] = int(v0)

        base = "sre_il" if method.startswith("il@") else method
        name, _, tau = method.partition("#tau")
        backend.sre_il = models.get(name, models["sre_il"])
        backend.target_min_prob = float(tau) if tau else None
        r = backend.select(frame, ref_mask, method=base, target_visible=None if seen else False, step=step)
        block = None
        if r["chosen_index"] >= 0 and r["chosen_mask"]:
            vals = ids[decode_mask(r["chosen_mask"]) > 0]
            vals = vals[vals >= 0]
            if vals.size:
                u, n = np.unique(vals, return_counts=True)
                block = ut.colors[int(u[np.argmax(n)])]
        optimal = (block == target and graspable) or (block in cost and not graspable and cost[block] <= v0)
        rec["steps"].append({"n_masks": r["num_objects"], "target_index": r["target_index"], "block": block,
                             "graspable": bool(graspable), "v0": int(v0), "optimal": bool(optimal)})
        if block == target:
            rec["success"], rec["outcome"] = bool(graspable), "success" if graspable else "blocked_target_grasp"
            break
        if block is None:
            continue                                   # a mask with no block behind it: a wasted step
        ut.remove(block)
        if not ut._target_ok():
            rec["outcome"] = "target_disturbed"
            break
    rec["n_steps"] = len(rec["steps"])
    return rec


def summarize(out_dir):
    rows = [json.loads(l) for l in open(Path(out_dir) / "episodes.jsonl") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    n = len({r["scene"] for r in rows})
    rng = np.random.RandomState(0)
    print(f"{n} twin scenes ({out_dir}); ideal removal, a grasp at a blocked target fails the episode\n")
    print(f"{'method':<34} {'success':>9} {'95% interval':>13} {'covered':>9} {'free':>8} {'optimal':>9} {'first opt':>10} "
          f"{'excess':>7}  outcomes")
    for m, eps in sorted(by.items(), key=lambda kv: -np.mean([r["success"] for r in kv[1]])):
        ok = np.array([r["success"] for r in eps])
        lo, hi = np.percentile([rng.choice(ok, len(ok)).mean() for _ in range(2000)], [2.5, 97.5])
        hard = [r["success"] for r in eps if r["optimal_steps"] > 1]
        easy = [r["success"] for r in eps if r["optimal_steps"] == 1]
        excess = [r["n_steps"] - r["optimal_steps"] for r in eps if r["success"]]
        best = sum(r["success"] and r["n_steps"] <= r["optimal_steps"] for r in eps)
        outc = defaultdict(int)
        for r in eps:
            outc[r["outcome"]] += 1
        print(f"{m:<34} {ok.sum():>4}/{len(ok):<4} {f'[{lo:.0%}, {hi:.0%}]':>13} "
              f"{f'{sum(hard)}/{len(hard)}':>9} {f'{sum(easy)}/{len(easy)}':>8} {f'{best}/{len(eps)}':>9} "
              f"{np.mean([r['steps'][0]['optimal'] for r in eps]):>10.0%} "
              f"{np.mean(excess) if excess else float('nan'):>7.2f}  {dict(outc)}")
    print("\ncovered / free = scenes where the search needs at least one removal / none; optimal = succeeded in the"
          "\nsearch's number of steps (removing blocks that are not in the way still succeeds, but slower); first opt ="
          "\nfirst choice has the minimum search cost; excess = steps beyond the optimum on successful episodes.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--out", default="save/twin_eval/run")
    ap.add_argument("--n_scenes", type=int, default=200)
    ap.add_argument("--layouts", default=None,
                    help="layouts.json of twin/make_eval_layouts.py: run its EVAL scenes (the ones built on the real "
                         "robot) instead of sampling; 'scene' in the records is then the layout id")
    ap.add_argument("--seed", type=int, default=12345, help="training used seeds below 10000")
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--methods", nargs="*", default=["sre_il", "sre", "heuristic", "random"])
    ap.add_argument("--taus", type=float, nargs="*", default=[],
                    help="also run every --il-ckpts model with backend.target_min_prob set to each of these")
    ap.add_argument("--il-ckpts", nargs="*", default=[])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return 0

    import torch
    from twin import augment
    from robot.backend import SpatialEncoder, UnveilerBackend
    from robot.replay_offline import _model_args
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = UnveilerBackend(device=args.device, warp=WARP, output_dir=str(out / "server_logs"), seed=args.seed)
    models = {"sre_il": backend.sre_il}
    for path in args.il_ckpts:
        m = SpatialEncoder(_model_args(backend)).to(backend.device)
        m.load_state_dict(torch.load(path, map_location=backend.device))
        models[f"il@{Path(path).parent.name}/{Path(path).stem}"] = m.eval()
    extra = [m for m in models if m != "sre_il"]
    methods = (list(args.methods) + extra + [f"{m}#tau{t}" for m in extra for t in args.taus]
               + ["always_target", "search"])

    ut = UnveilTwin(unveil_args([]))
    rng = np.random.default_rng(args.seed)
    fixed = [l for l in json.load(open(args.layouts))["layouts"] if l["role"] == "EVAL" and "targets" not in l] if args.layouts else None
    if fixed:
        args.n_scenes = len(fixed)
    done = 0
    with open(out / "episodes.jsonl", "a") as f:
        while done < args.n_scenes:
            if fixed:
                lay, target = fixed[done], fixed[done]["target"]
                state = (np.array(lay["qpos"]), lay["present"])
                ut.tw.episode_camera = dict(ut.tw.base_camera)
                ut.restore(state)
                for c in ut.present():                 # the reference photo: the target alone, as on the robot
                    if c != target:
                        ut.remove(c, settle=False)
                ref_ids = ut.block_ids(ut.tw.episode_camera)
                ut.restore(state)
            else:
                scene = ut.sample_scene(rng)
                if scene is None:
                    continue
                target, ref_ids = scene
            ref_mask = (ref_ids == ut.colors.index(target)).astype(np.uint8) * 255
            if np.count_nonzero(ref_mask) < MIN_TARGET_PX:
                continue
            state, aug, seed = ut.save(), augment.sample_params(ut.tw.cfg, rng), int(rng.integers(1 << 31))
            ut.start_search(target)
            if ut.value(()) > 4 and not fixed:         # the target was knocked away while the scene was built
                ut.restore(state)
                continue
            for m in methods:
                if m in ("always_target", "search"):   # no perception: the trivial baseline and the optimum
                    ut.restore(state)
                    ut.start_search(target)
                    v0, graspable = ut.value(()), ut._state(())[1]
                    ut.restore(state)
                    ok = bool(graspable) if m == "always_target" else True
                    rec = {"method": m, "success": ok, "optimal_steps": int(v0), "n_steps": 1 if m == "always_target"
                           else int(v0), "outcome": "success" if ok else "blocked_target_grasp",
                           "steps": [{"optimal": ok}]}
                else:
                    rec = run_episode(ut, backend, models, m, state, target, ref_mask, aug, seed, args.max_steps)
                rec.update(scene=fixed[done]["id"] if fixed else done, target=target, n_blocks=len(state[1]) and sum(state[1].values()))
                f.write(json.dumps(rec) + "\n")
                f.flush()
            done += 1
            if done % 20 == 0:
                print(f"{done}/{args.n_scenes} scenes", flush=True)
    summarize(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
