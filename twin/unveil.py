"""Covered-target scenes in the DOFBOT twin, labelled by search, for fine-tuning the SRE (real2sim training data).

The twin (twin/scene.py, from verify2act) renders the real setup: the same blocks, sheet, table and arm camera. Here
it makes the Task 1 scenes of docs/sre_training_and_real_eval.md: at most four blocks of different colours, with the
target covered from above by one or two of the others, leaned on, or free. Each render goes through the real robot's
preprocessing (robot/backend.py: the fixed workspace warp to the 400x400 view, then Mask R-CNN), so the SRE trains on
the tensors it gets on the robot.

Labels come from the same search as trainer/train_sre_exit.py, in MuJoCo: Q(s, i) = 1 + cost-to-go after lifting
block i out and letting the pile settle. The target is graspable from the top when no block covers more than
``max_cover`` of it (vertical rays) and both fingers of a parallel gripper have room beside it.

    python -m twin.unveil --preview 12 --out save/sre_twin/preview          # look at scenes before training
    python -m twin.unveil --out save/sre_twin --states 4000 --workers 3    # collect, then fine-tune the IL SRE
    python -m twin.unveil --out save/sre_twin2 --states 4000 --iterations 3    # expert iteration (AlphaZero style)

With --iterations > 1 this is the loop of trainer/train_sre_exit.py: iteration 0 rolls out the search's choices, later
iterations roll out the current SRE with probability 1 - beta, label every state it reaches with the search, add
them to the data and retrain.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SIM_SIZE = 400                                   # robot/backend.py
WARP = [88, 71, 501, 38, 639, 479, 54, 479]      # workspace corners in the real frame (scripts/replay_new_ckpts.sh)


def warp_matrix(warp=WARP):
    src = np.float32(warp).reshape(4, 2)
    dst = np.float32([[0, 0], [SIM_SIZE, 0], [SIM_SIZE, SIM_SIZE], [0, SIM_SIZE]])
    return cv2.getPerspectiveTransform(src, dst)


class UnveilTwin:
    """The twin plus what the SRE data needs: block-id images, the top-grasp test and the removal search."""

    def __init__(self, args, cfg=None):
        import mujoco
        from twin.config import load_config
        from twin.scene import DofbotTwin
        self.mj, self.a = mujoco, args
        self.tw = DofbotTwin(cfg or load_config(args.config))
        self.colors = self.tw.colors
        self.half = self.tw.size / 2
        self.H = warp_matrix()
        self.fail = args.max_depth + 3

    # ── state ──

    def save(self):
        return self.tw.data.qpos.copy(), dict(self.tw.present)

    def restore(self, state):
        qpos, present = state
        self.tw.data.qpos[:] = qpos
        self.tw.data.qvel[:] = 0.0
        self.tw.present = dict(present)
        self.mj.mj_forward(self.tw.model, self.tw.data)

    def present(self):
        return [c for c in self.colors if self.tw.present[c]]

    def remove(self, c, settle=True):
        """Lift a block out of the scene (ideal removal) and let the rest settle."""
        self.tw.present[c] = False
        self.tw._set_pose(c, self.tw._park_pos(self.colors.index(c)), 0.0)
        self.mj.mj_forward(self.tw.model, self.tw.data)
        if settle:
            self.tw.settle(self.a.settle_steps)

    def frame(self, c):
        b = self.tw._body[c]
        return self.tw.data.xpos[b].copy(), self.tw.data.xmat[b].reshape(3, 3).copy()

    # ── rendering ──

    def block_ids(self, cam):
        """(H, W) image: index into ``colors`` of the block seen at each pixel, -1 elsewhere."""
        tw = self.tw
        tw._set_camera(cam)
        tw.renderer.enable_segmentation_rendering()
        try:
            tw.renderer.update_scene(tw.data, camera="home")
            seg = tw.renderer.render()
        finally:
            tw.renderer.disable_segmentation_rendering()
        out = np.full(seg.shape[:2], -1, np.int16)
        for i, c in enumerate(self.colors):
            out[(seg[..., 0] == tw._geom[c]) & (seg[..., 1] == self.mj.mjtObj.mjOBJ_GEOM)] = i
        return out

    def to_sim(self, img, nearest=False):
        return cv2.warpPerspective(img, self.H, (SIM_SIZE, SIM_SIZE),
                                   flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)

    # ── can the target be grasped from the top? ──

    def coverage(self, t):
        """Share of the target, seen from straight above, that another block covers."""
        mj, tw = self.mj, self.tw
        p, R = self.frame(t)
        g = np.linspace(-0.8, 0.8, 6)
        geomid = np.zeros(1, np.int32)
        blocks = {tw._geom[c]: c for c in self.present()}
        seen = covered = 0
        for u in g:
            for v in g:
                q = p + R @ np.array([u * self.half[0], v * self.half[1], 0.0])
                mj.mj_ray(tw.model, tw.data, np.array([q[0], q[1], 0.6]), np.array([0.0, 0.0, -1.0]), None, 1, -1,
                          geomid)
                hit = blocks.get(int(geomid[0]))
                if hit is not None:
                    seen += 1
                    covered += hit != t
        return covered / max(seen, 1)

    def _inside_any(self, pts, others):
        for o in others:
            p, R = self.frame(o)
            local = (pts - p) @ R
            if (np.abs(local) <= self.half + 1e-4).all(axis=1).any():
                return True
        return False

    def fingers_ok(self, t):
        """Room for both fingers of a top-down parallel grasp across one of the target's 30 mm sides."""
        a = self.a
        p, R = self.frame(t)
        others = [c for c in self.present() if c != t]
        g = np.linspace(-0.5, 0.5, 3)
        for k in (1, 2):                                    # close across the block's y or z side
            axis = R[:, k]
            if abs(axis[2]) > 0.5:                          # that side points up: not a closing direction
                continue
            long_axis, up = R[:, 0], np.cross(R[:, 0], axis)
            free = True
            for s in (-1.0, 1.0):
                centre = p + s * axis * (self.half[k] + a.finger_gap + a.finger_len / 2)
                pts = np.array([centre + u * a.finger_len * axis + v * a.finger_width * long_axis
                                + w * 2 * self.half[k] * up for u in g for v in g for w in g])
                pts = pts[pts[:, 2] > 0.002]                # the table is not an obstacle to a finger tip above it
                if len(pts) and self._inside_any(pts, others):
                    free = False
                    break
            if free:
                return True
        return False

    def graspable(self, t):
        return self.coverage(t) <= self.a.max_cover and self.fingers_ok(t)

    # ── search ──

    def start_search(self, target):
        self.root, self.target, self.memo = self.save(), target, {}
        self.t0 = self.frame(target)[0]

    def _target_ok(self):
        p, _ = self.frame(self.target)
        return self.tw.present[self.target] and np.linalg.norm(p[:2] - self.t0[:2]) < 0.08 and p[2] > 0.0

    def _state(self, removed):
        self.restore(self.root)
        for c in removed:
            self.remove(c, settle=False)
        if removed:
            self.tw.settle(self.a.settle_steps)
        if not self._target_ok():
            return False, False, []
        return True, self.graspable(self.target), [c for c in self.present() if c != self.target]

    def value(self, removed=()):
        key = frozenset(removed)
        if key not in self.memo:
            ok, graspable, cands = self._state(removed)
            if not ok:
                v = self.fail
            elif graspable:
                v = 1
            elif len(removed) >= self.a.max_depth or not cands:
                v = self.a.max_depth + 2
            else:
                v = min(min(1 + self.value(tuple(removed) + (c,)) for c in cands), self.fail)
            self.memo[key] = v
        return self.memo[key]

    def q_values(self, mask_blocks):
        """Cost of choosing each segmented mask now (a mask with no block behind it wastes the step)."""
        v0 = self.value(())
        graspable = self._state(())[1]
        q = []
        for b in mask_blocks:
            if b is None:
                q.append(1 + v0)
            elif b == self.target:
                q.append(1 if graspable else 1 + v0)
            else:
                q.append(min(1 + self.value((b,)), self.fail))
        self.restore(self.root)
        return np.array(q, np.float32), v0, graspable

    # ── scenes ──

    def sample_scene(self, rng):
        """A layout of 2-4 blocks with a target that is covered, leaned on, or free. Returns (target, reference
        block-id image taken before the occluders went on), or None when the layout did not work out."""
        a, tw = self.a, self.tw
        n = int(rng.choice([2, 3, 4], p=a.n_blocks_p))
        blocks = [str(c) for c in rng.choice(self.colors, n, replace=False)]
        tw.randomize_episode(rng)
        try:
            tw.reset(rng, blocks)
        except RuntimeError:
            return None
        target = str(rng.choice(blocks))
        ref_ids = self.block_ids(tw.episode_camera)
        others = [c for c in blocks if c != target]
        if rng.random() < a.cover_prob:
            k = 2 if len(others) >= 2 and rng.random() < a.two_cover_prob else 1
            for c in rng.permutation(others)[:k]:
                p, yaw = tw.pose(target)
                d = rng.uniform(-a.cover_offset, a.cover_offset, 2) * self.half[:2]
                cs, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                x, y = p[0] + cs * d[0] - sn * d[1], p[1] + sn * d[0] + cs * d[1]
                tw._drop(str(c), x, y, yaw + rng.uniform(0, 180), rng)
        if not self._in_workspace(blocks):
            return None
        return target, ref_ids

    def _in_workspace(self, blocks):
        """Every block is on the table and its centre is inside the warped workspace view."""
        pts = np.array([self.tw.pose(c)[0] for c in blocks])
        if (pts[:, 2] < 0).any():
            return False
        uv = self.tw.project(pts)
        if np.isnan(uv).any():
            return False
        w = cv2.perspectiveTransform(uv.reshape(-1, 1, 2).astype(np.float32), self.H).reshape(-1, 2)
        return bool(((w > 15) & (w < SIM_SIZE - 15)).all())


def mask_blocks(masks, ids_sim, colors, min_agree=0.3):
    """Block behind each Mask R-CNN mask: the most common block id among its pixels in the warped id image."""
    out = []
    for m in masks:
        vals = ids_sim[m > 0]
        vals = vals[vals >= 0]
        if vals.size == 0:
            out.append(None)
            continue
        ids, counts = np.unique(vals, return_counts=True)
        out.append(colors[int(ids[np.argmax(counts)])] if counts.max() >= min_agree * np.count_nonzero(m) else None)
    return out


def observe(ut, seg, seg_dir, rng, aug):
    """One camera frame through the robot's preprocessing: (sim-view RGB, masks, bboxes, block behind each mask)."""
    from twin import augment
    tw = ut.tw
    cam = tw._jitter(tw.episode_camera, tw.cfg["camera"]["jitter_frame"], rng)
    ids = ut.block_ids(cam)
    rgb = augment.apply(tw.render(cam=cam), aug, rng)
    color = ut.to_sim(rgb)
    masks, _, _, bboxes = seg.from_maskrcnn(color, dir=seg_dir, bbox=True, dim=(SIM_SIZE, SIM_SIZE))
    return rgb, color, masks, bboxes, mask_blocks(masks, ut.to_sim(ids, nearest=True), ut.colors)


def collect(wid, args, n_states, out_dir, seed, preview=0, student_path=None, beta=1.0):
    import torch
    from mask_rg.object_segmenter import ObjectSegmenter
    from trainer.train_sre_exit import load_sre, sre_arrays
    from twin import augment

    rng = np.random.default_rng(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seg = ObjectSegmenter(Namespace(device=device, num_patches=args.num_patches, sequence_length=1))
    seg.threshold = args.seg_threshold
    seg_dir = tempfile.mkdtemp(prefix=f"twin_seg{wid}_")
    student = load_sre(student_path, device, args.num_patches).eval() if student_path and beta < 1 else None
    ut = UnveilTwin(args)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buf, shard, t0 = [], 0, time.time()
    stats = {"scenes": 0, "states": 0, "expert_actions": 0, "student_actions": 0, "no_layout": 0, "no_useful_mask": 0, "target_lost": 0, "v_hist": {},
             "masks": 0, "blocks": 0, "hidden_target": 0}

    def flush():
        nonlocal buf, shard
        if buf:
            np.savez_compressed(out_dir / f"w{wid}_{shard:03d}.npz",
                                **{k: np.stack([b[k] for b in buf]) for k in buf[0]})
            buf, shard = [], shard + 1

    while stats["states"] < n_states:
        scene = ut.sample_scene(rng)
        if scene is None:
            stats["no_layout"] += 1
            continue
        target, ref_ids = scene
        t_i = ut.colors.index(target)
        ref_mask = (ut.to_sim((ref_ids == t_i).astype(np.uint8) * 255, nearest=True) > 0).astype(np.uint8) * 255
        if np.count_nonzero(ref_mask) < 200:
            stats["no_layout"] += 1
            continue
        aug = augment.sample_params(ut.tw.cfg, rng)
        stats["scenes"] += 1

        for step in range(args.max_depth + 2):
            rgb, color, masks, bboxes, blocks = observe(ut, seg, seg_dir, rng, aug)
            k = min(len(masks), args.num_patches)
            if k == 0:
                break
            ut.start_search(target)
            q, v0, graspable = ut.q_values(blocks)
            if v0 >= ut.fail:
                stats["target_lost"] += 1
                break
            qk = q[:k]
            t_list = [i for i in range(k) if blocks[i] == target]
            t_idx = max(t_list, key=lambda i: np.count_nonzero(masks[i])) if t_list else -1
            # the robot's target mask is the reference photo's; half the time use the current mask when there is one
            tgt_mask = masks[t_idx] if t_idx >= 0 and rng.random() < 0.5 else ref_mask
            useful = qk.min() < 1 + v0 or (graspable and t_idx >= 0)
            if not useful:
                stats["no_useful_mask"] += 1            # the block to remove has no mask: nothing to learn here
            elif v0 > 1 or rng.random() < args.keep_easy:
                pi = np.exp(-(qk - qk.min()) / args.tau)
                pi /= pi.sum()
                s, tg, m, b, _ = sre_arrays(color, tgt_mask, masks, bboxes, args.num_patches)
                pi_pad, q_pad = np.zeros(args.num_patches, np.float32), np.full(args.num_patches, np.nan, np.float32)
                pi_pad[:k], q_pad[:k] = pi, qk
                buf.append({"scene": s, "target": tg, "masks": m, "bboxes": b, "n": np.int16(k), "pi": pi_pad,
                            "q": q_pad, "v0": np.float32(v0), "target_index": np.int16(t_idx),
                            "n_bodies": np.int16(len(ut.present()))})
                stats["states"] += 1
                stats["masks"] += k
                stats["blocks"] += len(ut.present())
                stats["hidden_target"] += t_idx < 0
                stats["v_hist"][int(v0)] = stats["v_hist"].get(int(v0), 0) + 1
                if stats["states"] <= preview:
                    cv2.imwrite(str(out_dir / f"w{wid}_{stats['states']:03d}.jpg"),
                                preview_panel(rgb, color, masks, tgt_mask, pi, qk, v0, target, blocks))
                if len(buf) >= 200:
                    flush()
            if student is not None and rng.random() >= beta:
                # the SRE's own choice (DAgger): grasping at the target or at a mask with no block ends the rollout
                s, tg, m, b, _ = sre_arrays(color, tgt_mask, masks, bboxes, args.num_patches)
                f32 = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=device)
                with torch.no_grad():
                    logits, _ = student(f32(s)[None, None], f32(tg)[None, None], f32(m)[None, :, None], f32(b)[None])
                chosen = blocks[int(torch.argmax(logits[0, :k]).item())]
                stats["student_actions"] += 1
                ut.restore(ut.root)
                if chosen is None or chosen == target:
                    break
                ut.remove(chosen)
                continue
            # expert action: the cheapest choice among the blocks (ties at random); stop once the target is free
            stats["expert_actions"] += 1
            if graspable:
                break
            cands = [c for c in ut.present() if c != target]
            if not cands:
                break
            cost = np.array([ut.value((c,)) for c in cands])
            ut.restore(ut.root)
            ut.remove(str(rng.choice([c for c, v in zip(cands, cost) if v == cost.min()])))

        if stats["scenes"] % 50 == 0:
            print(f"[w{wid}] {stats['states']}/{n_states} states, {stats['scenes']} scenes, "
                  f"{stats['states'] / max(time.time() - t0, 1):.2f} states/s, cost-to-go "
                  f"{dict(sorted(stats['v_hist'].items()))}, no useful mask {stats['no_useful_mask']}", flush=True)
    flush()
    with open(out_dir / f"w{wid}_stats.json", "w") as f:
        json.dump(stats, f, default=int)
    ut.tw.close()


def preview_panel(rgb, color, masks, tgt_mask, pi, q, v0, target, blocks):
    """Camera frame next to the SRE's view, with masks, the target (red) and the label on each mask."""
    sim = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    for i, m in enumerate(masks[:len(pi)]):
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sim, cs, -1, (0, 215, 255), 1)
        ys, xs = np.nonzero(m)
        txt = f"{i}:{pi[i]:.2f} q{q[i]:.0f}"
        cv2.putText(sim, txt, (int(xs.mean()) - 30, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(sim, txt, (int(xs.mean()) - 30, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cs, _ = cv2.findContours((tgt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sim, cs, -1, (0, 0, 255), 2)
    cv2.putText(sim, f"target {target}, optimum {v0:.0f} step(s)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1)
    frame = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (int(640 * SIM_SIZE / 480), SIM_SIZE))
    return np.hstack([frame, sim])


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="save/sre_twin")
    ap.add_argument("--states", type=int, default=4000, help="states collected per iteration")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--beta_decay", type=float, default=0.5, help="P(expert action) = decay**iteration")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--preview", type=int, default=0, help="write this many labelled scenes as JPEGs and exit")
    ap.add_argument("--init", default="save/sre/sre_model_best.pt", help="SRE weights to fine-tune (the IL model)")
    ap.add_argument("--config", default=None, help="twin YAML (default twin/dofbot_twin.yaml)")
    ap.add_argument("--n_blocks_p", type=float, nargs=3, default=[0.2, 0.35, 0.45], help="P(2, 3, 4 blocks)")
    ap.add_argument("--cover_prob", type=float, default=0.8, help="share of scenes with a block dropped on the target")
    ap.add_argument("--two_cover_prob", type=float, default=0.3)
    ap.add_argument("--cover_offset", type=float, default=0.9, help="drop offset, in target half-extents")
    ap.add_argument("--max_cover", type=float, default=0.1, help="max share of the target covered from above")
    ap.add_argument("--finger_len", type=float, default=0.01, help="m, finger slot depth beside the target")
    ap.add_argument("--finger_width", type=float, default=0.02, help="m, finger slot width along the block")
    ap.add_argument("--finger_gap", type=float, default=0.002)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--settle_steps", type=int, default=250)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--keep_easy", type=float, default=0.35)
    ap.add_argument("--seg_threshold", type=float, default=0.97, help="as robot/backend.py")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_patches", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.preview:
        collect(0, args, args.preview, out, args.seed, preview=args.preview)
        return 0

    import glob
    import multiprocessing as mp
    from trainer.train_sre_exit import load_shards, train_student
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    logf = open(out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    ctx = mp.get_context("spawn")
    student = None
    for it in range(args.iterations):
        beta = args.beta_decay ** it
        data = out / f"it{it}"
        have = len(load_shards([data])["n"]) if glob.glob(str(data / "w*_*.npz")) else 0
        need = args.states - have
        log(f"iteration {it}: beta {beta:.2f}, student {student}, {have} states already collected")
        if need > 0:
            per = [need // args.workers + (w < need % args.workers) for w in range(args.workers)]
            # worker ids continue after the shards already there, so a rerun adds to them
            first = len({Path(f).name.split("_")[0] for f in glob.glob(str(data / "w*_*.npz"))})
            procs = [ctx.Process(target=collect, args=(first + w, args, per[w], data,
                                                       args.seed * 1000 + it * 100 + first + w, 0, student, beta),
                                 daemon=True) for w in range(args.workers)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()
        ckpt = out / f"sre_exit_it{it}.pt"
        res = train_student(args, [out / f"it{j}" for j in range(it + 1)], args.init, ckpt, log)
        log(f"iteration {it}: trained on {res['states']} states, best val loss {res['val_loss']:.3f} -> {ckpt}")
        student = str(ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
