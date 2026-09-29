"""Expert-iteration training for the SRE (replaces the PPO fine-tuning in train_sre_rl.py).

Why not PPO: each PPO sample was a full physics rollout whose return was dominated by Action-Decoder grasp noise
(50-70% of grasps fail regardless of the choice), the critic only saw the actor's own detached logits, and a 0.05
entropy bonus on top of those near-zero advantages drove the policy to uniform (measured: top-1 probability 0.41 at
episode 1000 -> 0.25 at 9000 on 4-object scenes).

What this does instead (Expert Iteration / approximate policy iteration with the simulator as the model):

    MDP     state = the scene; action = which segmented object to remove next (the target itself = grasp it now);
            cost = 1 per action; the episode ends when the target is grasped. Removal is ideal (the body is lifted
            out and physics settles), because selection is the SRE's job and execution is the Action Decoder's.
    Expert  exact search in PyBullet. p.saveState/restoreState lets us try every candidate removal, let the pile
            settle (so knock-on collapses and a toppled target count), and test whether the target is graspable:
            a gripper-sized corridor from the target outwards must be free in at least one of 16 approach directions
            (the Action Decoder's 16 push-grasp angles; --approach_cone restricts them, e.g. to a fixed arm's reach).
            Q(s, i) = 1 + cost-to-go after removing i, by depth-limited search with memoisation.
    Student the SRE, trained with soft cross-entropy to softmax(-Q / tau) on the Mask R-CNN masks it sees at test time.
    Loop    iteration 0 rolls out the expert; later iterations roll out the student with probability 1 - beta and label
            every visited state with the expert (DAgger), aggregate, retrain.

No hand-written ranking rule appears anywhere: the labels come from simulated consequences, which is what the
paper's accessibility and stability claims need.

Usage:
    # 0. check that the corridor test predicts real Action-Decoder grasps (worth doing once)
    python -m trainer.train_sre_exit --probe 30
    # 1. train (4 iterations x 3000 states, 3 collection workers; about 3-4 h on one GPU)
    python -m trainer.train_sre_exit --out save/sre_exit --iterations 4 --states_per_iter 3000 --workers 3
    # 2. evaluate like any SRE-IL checkpoint (same architecture):
    python eval_selectors.py --selectors oracle sre_il heuristic --sre_model save/sre_exit/sre_exit_best.pt ...
    python -m robot.server ... --sre-il-ckpt save/sre_exit/sre_exit_best.pt     # method `sre_il` on the robot
"""

import argparse
import glob
import json
import math
import os
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── geometry: is the target graspable, and what does a removal do ────────────────────────────────────────────────────

class Probe:
    """A static, massless box used as the gripper's approach corridor for collision queries only."""

    FAR = [50.0, 50.0, 50.0]

    def __init__(self, p, length, width, height):
        self.p, self.length, self.width, self.height = p, length, width, height
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length / 2, width / 2, height / 2])
        self.body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=self.FAR)
        p.setCollisionFilterGroupMask(self.body, -1, 0, 0)     # never pushes anything during stepping

    def free_directions(self, target, others, dirs, gap=0.005, z0=0.005):
        """Approach angles (radians, world frame) whose corridor to the target touches no other object, and the set
        of objects blocking at least one corridor."""
        p = self.p
        lo, hi = p.getAABB(target)
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        hx, hy = (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2
        free, blockers = [], set()
        for th in dirs:
            c, s = math.cos(th), math.sin(th)
            r = abs(hx * c) + abs(hy * s)                       # AABB support distance in this direction
            d = r + gap + self.length / 2
            pos = [cx + c * d, cy + s * d, z0 + self.height / 2]
            p.resetBasePositionAndOrientation(self.body, pos, p.getQuaternionFromEuler([0, 0, th]))
            hit = [o for o in others if p.getClosestPoints(bodyA=self.body, bodyB=o, distance=0.0)]
            blockers.update(hit)
            if not hit:
                free.append(th)
        p.resetBasePositionAndOrientation(self.body, self.FAR, [0, 0, 0, 1])
        return free, blockers


def top_coverage(p, target, others, grid=6, shrink=0.8):
    """Share of the target's footprint covered from above by other objects (vertical rays over its AABB)."""
    lo, hi = p.getAABB(target)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    hx, hy = shrink * (hi[0] - lo[0]) / 2, shrink * (hi[1] - lo[1]) / 2
    xs, ys = np.linspace(cx - hx, cx + hx, grid), np.linspace(cy - hy, cy + hy, grid)
    hits = [h[0] for h in p.rayTestBatch([[x, y, 1.0] for x in xs for y in ys],
                                         [[x, y, -0.02] for x in xs for y in ys])]
    others = set(others)
    first = [b for b in hits if b == target or b in others]      # rays whose first hit is an object
    return sum(b != target for b in first) / max(len(first), 1)


class Search:
    """Exact cost-to-go by depth-limited search over removals from a saved PyBullet state.

    access='side': graspable = enough free horizontal approach corridors (the sim Barrett push-grasp).
    access='top' : graspable = nothing lies on the target (vertical rays) and a parallel gripper has room for both
                   fingers on some grasp axis (the real DOFBOT's top-down grasp)."""

    def __init__(self, p, probe, args):
        self.p, self.probe, self.a = p, probe, args
        n = args.n_dirs
        dirs = [2 * math.pi * k / n for k in range(n)]
        if args.approach_cone:
            c, hw = math.radians(args.approach_cone[0]), math.radians(args.approach_cone[1])
            dirs = [t for t in dirs if abs((t - c + math.pi) % (2 * math.pi) - math.pi) <= hw]
        self.dirs = dirs
        # graspable = enough free approach directions for the Action Decoder to find one (probe: with 16 directions,
        # targets with >= 7 free were grasped 45% of the time, with < 7 never)
        self.min_free = max(1, math.ceil(args.min_free_frac * len(dirs)))
        self.fail = args.max_depth + 3              # cost of a lost / toppled target

    def reset(self, root_state, target, objects):
        self.root, self.target, self.objects = root_state, target, list(objects)
        self.memo = {}
        lo, hi = self.p.getAABB(target)
        self.t_pos0 = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        self.nodes = 0

    def _apply(self, removed):
        p = self.p
        p.restoreState(self.root)
        for k, b in enumerate(removed):          # lift the removed objects out of the scene
            p.resetBasePositionAndOrientation(b, [20.0 + 0.5 * k, 20.0, -0.6], [0, 0, 0, 1])
            p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0])
        if removed:
            for _ in range(self.a.settle_steps):
                p.stepSimulation()

    def _target_ok(self):
        p = self.p
        pos, q = p.getBasePositionAndOrientation(self.target)
        rz = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)[:, 2]
        tilt = math.acos(max(-1.0, min(1.0, rz[2])))
        lo, _ = p.getAABB(self.target)
        moved = np.linalg.norm(np.array([pos[0], pos[1]]) - self.t_pos0[:2])
        return tilt < self.a.max_tilt and lo[2] > -0.02 and moved < 0.08

    def _state(self, removed):
        """(target ok, graspable now, candidate removals) after `removed`. Candidates = objects blocking any
        approach corridor or touching the target; removing anything else cannot change graspability directly."""
        self._apply(removed)
        self.nodes += 1
        if not self._target_ok():
            return False, False, []
        p, t = self.p, self.target
        left = [o for o in self.objects if o not in removed and o != t
                and p.getBasePositionAndOrientation(o)[0][2] > -0.05]
        touching = {o for o in left if p.getClosestPoints(bodyA=o, bodyB=t, distance=self.a.near_dist)}
        if self.a.access == "top":
            covered = top_coverage(p, t, left) > self.a.max_cover
            free, blockers = self.probe.free_directions(t, left, self.dirs)     # finger slots around the target
            n = self.a.n_dirs
            free_k = {int(round(f / (2 * math.pi / n))) % n for f in free}
            fingers_ok = any((k + n // 2) % n in free_k for k in free_k)        # both fingers of one grasp axis
            return True, (not covered) and fingers_ok, sorted(blockers | touching)
        free, blockers = self.probe.free_directions(t, left, self.dirs)
        return True, len(free) >= self.min_free, sorted(blockers | touching)

    def value(self, removed=()):
        """Minimum number of actions (removals + the final target grasp) from the state after `removed`."""
        key = frozenset(removed)
        if key in self.memo:
            return self.memo[key]
        ok, graspable, cands = self._state(removed)
        if not ok:
            v = self.fail
        elif graspable:
            v = 1
        elif len(removed) >= self.a.max_depth:
            v = self.a.max_depth + 2            # unresolved within the horizon
        else:
            v = min([1 + self.value(tuple(removed) + (c,)) for c in cands], default=self.a.max_depth + 2)
            v = min(v, self.fail)
        self.memo[key] = v
        return v

    def q_values(self, mask_bodies):
        """Q for each segmented mask (cost of choosing it now). Masks with no body / far objects waste a step."""
        v0 = self.value(())
        ok, graspable, cands = self._state(())
        near = set(cands) if ok else set()
        q = []
        for b in mask_bodies:
            if b is None or not ok:
                q.append(1 + v0)
            elif b == self.target:
                q.append(1 if graspable else 1 + v0)
            elif b in near:
                q.append(min(1 + self.value((b,)), self.fail))
            else:
                q.append(1 + v0)
        self.p.restoreState(self.root)
        return np.array(q, dtype=np.float32), v0, graspable


def make_probe(p, args):
    """Side mode: the push-grasp approach corridor. Top mode: one finger slot beside the target."""
    if args.access == "top":
        return Probe(p, args.finger_len, args.finger_width, args.finger_height)
    return Probe(p, args.corridor_len, args.corridor_width, args.corridor_height)


def stack_on_target(p, target, others, rng, args):
    """Top-occlusion scenes: drop 1-2 of the target's nearest neighbours onto it and let them settle."""
    lo, hi = p.getAABB(target)
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
    near = sorted(others, key=lambda o: np.linalg.norm(np.array(p.getBasePositionAndOrientation(o)[0][:2]) - c))
    z = hi[2]
    for o in near[:rng.randint(1, 3)]:
        olo, ohi = p.getAABB(o)
        jitter = rng.uniform(-0.6, 0.6, 2) * np.array([hi[0] - lo[0], hi[1] - lo[1]]) / 2
        pos = [c[0] + jitter[0], c[1] + jitter[1], z + (ohi[2] - olo[2]) / 2 + 0.01]
        p.resetBasePositionAndOrientation(o, pos, p.getQuaternionFromEuler([0, 0, rng.uniform(0, math.pi)]))
        p.resetBaseVelocity(o, [0, 0, 0], [0, 0, 0])
        z = pos[2] + (ohi[2] - olo[2]) / 2
    for _ in range(2 * args.settle_steps):
        p.stepSimulation()


# ── SRE inputs (identical to robot/backend.py and eval_selectors.py) ─────────────────────────────────────────────────

def sre_arrays(color, target_mask, masks, bboxes, num_patches):
    import utils.general_utils as gu
    k = min(len(masks), num_patches)
    m = np.zeros((num_patches, 100, 100), np.float16)
    b = np.zeros((num_patches, 4), np.float32)
    for i in range(k):
        m[i] = gu.resize_mask(masks[i])
        b[i] = gu.resize_bbox(bboxes[i])
    return (gu.resize_mask(color).mean(axis=2).astype(np.float16), gu.resize_mask(target_mask).astype(np.float16),
            m, b, k)


def load_sre(path, device, num_patches):
    import torch
    from policy.sre_model import SpatialEncoder
    model = SpatialEncoder(Namespace(device=device, num_patches=num_patches, sequence_length=1)).to(device)
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    return model


# ── collection worker ────────────────────────────────────────────────────────────────────────────────────────────────

def collect(wid, args, n_states, student_path, beta, out_dir, seed):
    import torch
    import yaml
    import pybullet as p
    from env.environment import Environment
    from mask_rg.object_segmenter import ObjectSegmenter
    import utils.general_utils as gu
    from eval_selectors import mask_bodies

    rng = np.random.RandomState(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        params = yaml.safe_load(f)
    env = Environment(params, objects_set=args.objects_set, render=args.render)
    seg = ObjectSegmenter(Namespace(device=device, num_patches=args.num_patches, sequence_length=1))
    seg_dir = tempfile.mkdtemp(prefix=f"exit_seg{wid}_")
    student = load_sre(student_path, device, args.num_patches).eval() if student_path and beta < 1 else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buf, shard, n_done, t0 = [], 0, 0, time.time()
    stats = {"scenes": 0, "states": 0, "expert_actions": 0, "student_actions": 0, "search_nodes": 0,
             "v_hist": {}}

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        np.savez_compressed(out_dir / f"w{wid}_{shard:03d}.npz",
                            **{k: np.stack([b[k] for b in buf]) for k in buf[0]})
        buf, shard = [], shard + 1

    while n_done < n_states:
        lo, hi = args.densities[rng.randint(len(args.densities))]
        env.nr_objects = [lo, hi]
        env.seed(int(rng.randint(0, 2 ** 31 - 1)))
        obs = env.reset()
        probe = make_probe(p, args)                     # env.reset cleared all bodies
        search = Search(p, probe, args)
        stats["scenes"] += 1

        masks, _, _, bboxes = seg.from_maskrcnn(obs['color'][1], dir=seg_dir, bbox=True)
        objects = [o.body_id for o in env.objects]
        bodies = mask_bodies(masks, obs['seg'][1], objects)
        with_body = [i for i, b in enumerate(bodies) if b is not None]
        if len(with_body) < 2:
            continue
        if rng.rand() < args.central_target:
            _, t_idx = gu.get_target_mask(masks, obs['color'][1], rng)
        else:
            t_idx = int(rng.choice(with_body))
        target = bodies[t_idx]
        if target is None:
            continue
        ref_mask = masks[t_idx]                         # the real protocol photographs the target alone first
        if args.access == "top" and rng.rand() < args.stack_prob:
            stack_on_target(p, target, [b for b in objects if b != target], rng, args)
            obs = env.get_observation()
            stats["stacked"] = stats.get("stacked", 0) + 1

        for step in range(args.max_depth + 3):
            objects = [o.body_id for o in env.objects]
            if target not in objects:
                break
            if step > 0 or args.access == "top":
                masks, _, _, bboxes = seg.from_maskrcnn(obs['color'][1], dir=seg_dir, bbox=True)
                bodies = mask_bodies(masks, obs['seg'][1], objects)
            t_list = [i for i, b in enumerate(bodies) if b == target]
            if t_list:
                t_idx = max(t_list, key=lambda i: np.count_nonzero(masks[i]))
                ref_mask = masks[t_idx]
            elif args.access == "top":
                t_idx = -1                              # hidden under other objects: use the reference mask
            else:
                break                                   # side mode: target no longer visible from above
            if t_idx >= args.num_patches:
                break

            root = p.saveState()
            search.reset(root, target, objects)
            q, v0, graspable = search.q_values(bodies)
            stats["search_nodes"] += search.nodes
            k = min(len(masks), args.num_patches)
            if v0 >= search.fail or k == 0:
                p.removeState(root)
                break
            qk = q[:k]
            pi = np.exp(-(qk - qk.min()) / args.tau)
            pi /= pi.sum()
            scene, tgt, m, b, _ = sre_arrays(obs['color'][1], ref_mask, masks, bboxes, args.num_patches)
            pi_pad = np.zeros(args.num_patches, np.float32)
            pi_pad[:k] = pi
            q_pad = np.full(args.num_patches, np.nan, np.float32)
            q_pad[:k] = qk
            keep = v0 > 1 or rng.rand() < args.keep_easy     # "grasp the target now" states are half of all
            if keep:
                buf.append({"scene": scene, "target": tgt, "masks": m, "bboxes": b, "n": np.int16(k), "pi": pi_pad,
                            "q": q_pad, "v0": np.float32(v0), "target_index": np.int16(t_idx),
                            "n_bodies": np.int16(len(objects))})
                n_done += 1
                stats["states"] += 1
                stats["v_hist"][int(v0)] = stats["v_hist"].get(int(v0), 0) + 1
            if len(buf) >= 200:
                flush()

            # choose the action to execute: expert (argmin Q, random tie-break) or the student (DAgger)
            if student is None or rng.rand() < beta:
                best = np.flatnonzero(qk == qk.min())
                a = int(rng.choice(best))
                stats["expert_actions"] += 1
            else:
                f32 = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=device)
                with torch.no_grad():   # same shapes as train_student's batches: [1,1,H,W], [1,N,1,H,W], [1,N,4]
                    logits, _ = student(f32(scene)[None, None], f32(tgt)[None, None], f32(m)[None, :, None],
                                        f32(b)[None])
                a = int(torch.argmax(logits[0, :k]).item())
                stats["student_actions"] += 1
            chosen = bodies[a]
            p.restoreState(root)
            p.removeState(root)
            if chosen == target:
                break                                   # grasping the target ends the episode either way
            if chosen is not None:
                p.resetBasePositionAndOrientation(chosen, [20.0, 20.0, -0.6], [0, 0, 0, 1])
                for _ in range(args.settle_steps):
                    p.stepSimulation()
            if args.access == "side":
                env.remove_flat_objs()                  # top mode keeps tilted objects: they lean on the target
            obs = env.get_observation()

        if stats["scenes"] % 10 == 0:
            rate = stats["states"] / max(time.time() - t0, 1)
            print(f"[w{wid}] {stats['states']}/{n_states} states, {stats['scenes']} scenes, {rate:.2f} states/s, "
                  f"cost-to-go histogram {dict(sorted(stats['v_hist'].items()))}", flush=True)
    flush()
    with open(out_dir / f"w{wid}_stats.json", "w") as f:
        json.dump(stats, f)


# ── student training ─────────────────────────────────────────────────────────────────────────────────────────────────

def load_shards(dirs):
    arrays = {}
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "w*_*.npz"))):
            z = np.load(f)
            for k in z.files:
                arrays.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in arrays.items()}


def train_student(args, data_dirs, init_path, out_path, log):
    import torch
    import torch.nn.functional as F

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_shards(data_dirs)
    n = len(data["n"])
    if n == 0:
        raise RuntimeError(f"no training states found in {data_dirs}")
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    model = load_sre(init_path, device, args.num_patches)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # collection keeps only `keep_easy` of the "target already graspable" states (cost-to-go 1); weight them back up
    # so the loss sees the true state distribution (without this the student under-picks the free target)
    weight = np.where(data["v0"] <= 1, 1.0 / max(args.keep_easy, 1e-3), 1.0).astype(np.float32)

    def batch(ids):
        t = lambda k, dt=torch.float32: torch.as_tensor(data[k][ids].astype(np.float32), dtype=dt, device=device)
        return (t("scene").unsqueeze(1), t("target").unsqueeze(1), t("masks").unsqueeze(2), t("bboxes"),
                t("pi"), torch.as_tensor(data["n"][ids].astype(np.int64), device=device),
                torch.as_tensor(np.nan_to_num(data["q"][ids], nan=1e3), device=device),
                torch.as_tensor(weight[ids], device=device))

    def run(ids, train):
        model.train(train)
        tot, n_b, hit, regret = 0.0, 0, 0, 0.0
        for s in range(0, len(ids), args.batch_size):
            scene, tgt, m, b, pi, k, q, w = batch(ids[s:s + args.batch_size])
            with torch.set_grad_enabled(train):
                logits, _ = model(scene, tgt, m, b)
                valid = torch.arange(args.num_patches, device=device)[None] < k[:, None]
                logp = F.log_softmax(logits.masked_fill(~valid, -1e4), dim=1)
                loss = (-(pi * logp).sum(1) * w).sum() / w.sum()
                if train:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            a = logits.masked_fill(~valid, -1e4).argmax(1)
            qa = q.gather(1, a[:, None])[:, 0]
            qmin = q.min(1).values
            hit += int((qa <= qmin + 1e-6).sum())
            regret += float((qa - qmin).sum())
            tot += float(loss) * len(k)
            n_b += len(k)
        return tot / n_b, hit / n_b, regret / n_b

    best = float("inf")
    for ep in range(args.epochs):
        rng.shuffle(tr_idx)
        tr = run(tr_idx, True)
        va = run(val_idx, False)
        log(f"  epoch {ep:2d}  train loss {tr[0]:.3f} opt {tr[1]:.1%}  |  val loss {va[0]:.3f} "
            f"optimal-choice {va[1]:.1%} regret {va[2]:.3f} steps")
        if va[0] < best:
            best = va[0]
            torch.save(model.state_dict(), out_path)
    return {"states": int(n), "val_loss": best}


# ── probe: does the corridor test predict what the Action Decoder can actually grasp? ────────────────────────────────

def probe_mode(args):
    import pybullet as p
    from eval_selectors import Harness, mask_bodies
    import utils.general_utils as gu

    h = Harness(Namespace(**{**vars(args), "selectors": [], "mask_noise": None, "nr_objects": [4, 10],
                             "ae_model": args.ae_model, "sre_model": None, "sre_rl": None}))
    rows = []
    rng = np.random.RandomState(args.seed)
    for k in range(args.probe):
        h.env.seed(int(rng.randint(0, 2 ** 31 - 1)))
        obs = h.env.reset()
        objects = [o.body_id for o in h.env.objects]
        masks, bboxes = h._segment(obs)
        bodies = mask_bodies(masks, obs['seg'][1], objects)
        cands = [i for i, b in enumerate(bodies) if b is not None]
        if not cands:
            continue
        t = int(rng.choice(cands))
        probe = make_probe(p, args)
        search = Search(p, probe, args)
        free, _ = probe.free_directions(bodies[t], [o for o in objects if o != bodies[t]], search.dirs)
        _, info = h.env.step(h._action3d(h._action_decoder(obs, masks[t])))
        for _ in range(240):
            p.stepSimulation()
        h.env.remove_flat_objs()
        after = [o.body_id for o in h.env.objects]
        grasped = bool(info["stable"]) and bodies[t] not in after
        rows.append((len(free) >= search.min_free, grasped))
        print(f"scene {k:2d}: free directions {len(free):2d}/{len(search.dirs)}  AD grasp of the target "
              f"{'succeeded' if grasped else 'failed'}", flush=True)
    rows = np.array(rows, dtype=bool)
    if len(rows):
        for acc in (True, False):
            sel = rows[:, 0] == acc
            if sel.any():
                print(f"corridor says {'graspable' if acc else 'blocked  '}: {sel.sum():3d} scenes, AD success "
                      f"{rows[sel, 1].mean():.0%}")
        print("A useful test separates these two rates clearly; tune --corridor_width/--corridor_len if not.")


# ── main ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="save/sre_exit")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--states_per_iter", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--beta0", type=float, default=1.0, help="P(expert action) in iteration 0")
    ap.add_argument("--beta_decay", type=float, default=0.5, help="beta = beta0 * decay**iteration")
    ap.add_argument("--init", default="save/sre/sre_model_best.pt",
                    help="student initialisation: an SRE state_dict (default: the IL SRE) or 'scratch'")
    ap.add_argument("--densities", type=lambda s: [tuple(map(int, r.split("-"))) for r in s.split(",")],
                    default=[(3, 6), (6, 9), (9, 13)], help="object-count ranges [lo, hi), e.g. 3-6,6-9,9-13")
    ap.add_argument("--central_target", type=float, default=0.5,
                    help="share of episodes whose target is the central object (as in eval); the rest random")
    # search / expert
    ap.add_argument("--max_depth", type=int, default=3, help="removals searched before giving up")
    ap.add_argument("--near_dist", type=float, default=0.01,
                    help="m; objects touching the target are search candidates (as are all corridor blockers)")
    ap.add_argument("--n_dirs", type=int, default=16, help="approach directions (the AD's 16 rotations)")
    ap.add_argument("--approach_cone", type=float, nargs=2, default=None, metavar=("CENTER_DEG", "HALF_WIDTH_DEG"),
                    help="only these world-frame approach directions count (a fixed arm's reach)")
    ap.add_argument("--access", default="side", choices=["side", "top"],
                    help="side: sim push-grasp from the side; top: top-down parallel grasp (the real DOFBOT)")
    ap.add_argument("--stack_prob", type=float, default=0.6, help="top mode: share of scenes stacked on the target")
    ap.add_argument("--max_cover", type=float, default=0.1, help="top mode: max share of the target covered")
    ap.add_argument("--finger_len", type=float, default=0.02, help="m, top mode: finger slot depth")
    ap.add_argument("--finger_width", type=float, default=0.03, help="m, top mode: finger slot width")
    ap.add_argument("--finger_height", type=float, default=0.03, help="m, top mode: finger slot height")
    ap.add_argument("--min_free_frac", type=float, default=0.4,
                    help="share of approach directions that must be free for the target to count as graspable")
    ap.add_argument("--corridor_len", type=float, default=0.12, help="m, gripper approach corridor length")
    ap.add_argument("--corridor_width", type=float, default=0.09, help="m, corridor width (open-hand span)")
    ap.add_argument("--corridor_height", type=float, default=0.08, help="m, corridor height above the table")
    ap.add_argument("--settle_steps", type=int, default=150)
    ap.add_argument("--max_tilt", type=float, default=0.5, help="rad; a target tilted more has toppled")
    ap.add_argument("--tau", type=float, default=0.3, help="soft-label temperature over Q (steps)")
    ap.add_argument("--keep_easy", type=float, default=0.35,
                    help="share of already-graspable states (cost-to-go 1) kept in the dataset")
    # student
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    # env / misc
    ap.add_argument("--render", default="egl", choices=["gui", "direct", "egl"])
    ap.add_argument("--objects_set", default="seen", help="train on 'seen'; eval_selectors defaults to 'unseen'")
    ap.add_argument("--config", default="yaml/bhand.yml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_patches", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ae_model", default="save/ae/ae_model_best.pt", help="only for --probe")
    ap.add_argument("--probe", type=int, default=0, help="check the corridor test against N real AD grasps and exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.probe:
        probe_mode(args)
        return 0

    import multiprocessing as mp
    import signal
    # turn SIGTERM (kill/pkill) into a normal exit, so multiprocessing terminates the daemon workers with us
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2, default=str)
    logf = open(out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    init = None if args.init == "scratch" else args.init
    student = init
    ctx = mp.get_context("spawn")
    for it in range(args.iterations):
        beta = args.beta0 * args.beta_decay ** it
        it_dir = out / f"it{it}"
        have = len(load_shards([it_dir])["n"]) if it_dir.exists() and glob.glob(str(it_dir / "w*_*.npz")) else 0
        need = args.states_per_iter - have
        log(f"iteration {it}: beta {beta:.2f}, student {student}, {have} states already collected")
        if need > 0:
            per = [need // args.workers + (w < need % args.workers) for w in range(args.workers)]
            procs = [ctx.Process(target=collect, args=(w, args, per[w], student, beta, it_dir,
                                                       args.seed * 1000 + it * 100 + w + have),
                                 daemon=True)          # workers die with the parent (no orphans holding the GPU)
                     for w in range(args.workers)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()
        ckpt = out / f"sre_exit_it{it}.pt"
        res = train_student(args, [out / f"it{j}" for j in range(it + 1)], init, ckpt, log)
        log(f"iteration {it}: trained on {res['states']} states, best val loss {res['val_loss']:.3f} -> {ckpt}")
        student = str(ckpt)
        import shutil
        shutil.copy(ckpt, out / "sre_exit_best.pt")
    log(f"done: {out / 'sre_exit_best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
