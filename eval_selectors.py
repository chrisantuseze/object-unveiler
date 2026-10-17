"""Selector harness: the same scenes, segmentation and Action Decoder for every object-selection method.

Unveiler factors a step into "which object next" (the selector) and "how to grasp it" (the Action Decoder). This
harness holds everything but the selector fixed, so differences in success come from selection alone, which is the
comparison the IROS AE asked for. It runs headless and labels outcomes from simulator ground truth (no prompts), so it
can run unattended on a cluster.

Selectors (all return an index into the segmented objects; the target's own index = grasp the target now):
    sre        SRE after PPO fine-tuning (full Unveiler)          sre_il     SRE after imitation only
    heuristic  Algorithm 1, the IL supervisor                     planner    occlusion-graph planner (classical)
    nearest    closest object to the target                       random     uniform over segmented objects
    clip       zero-shot CLIP (baseline/clip_eval.py)             gpt4o      GPT-4o (baseline/gpt.py)
    oracle     removes an object that physically touches the target (simulator ground truth); upper bound
    search     argmin Q of the expert-iteration search (--execution ideal only); the optimum of that MDP

Per step it also logs, from PyBullet ground truth:
    sel_valid   the chosen object touches the target (or the target is free and was chosen)  -> eps_SRE = 1 - mean
    exec_ok     a valid chosen object was actually removed by the grasp                      -> eps_exec = 1 - mean
    agrees_heuristic, probs (SRE), mask noise applied, timing.

Run (one density bin, all local selectors, headless GPU rendering):
    python eval_selectors.py --nr_objects 6 9 --n_scenes 30 --render egl --out save/selector_eval/6_9 \
        --selectors oracle sre sre_il heuristic planner nearest random
Summarize:
    python eval_selectors.py --summarize save/selector_eval/6_9
Selection only (--execution ideal): the chosen object is lifted out and the pile settles, and the target is retrieved
when the search's graspability test passes, exactly the MDP of trainer/train_sre_exit.py. The Action Decoder fails
60-90% of valid grasps, which hides selection in the default mode; here every step that fails is the selector's.
    python eval_selectors.py --execution ideal --nr_objects 6 9 --n_scenes 30 --out save/selector_eval/6_9_ideal \
        --selectors search oracle sre sre_il heuristic planner nearest random
Mask-noise sweep (Item 3): add e.g. --mask_noise merge:0.3  (types: merge, split, drop, erode, dilate)

Episodes are appended to <out>/episodes.jsonl as they finish, and (episode, selector) pairs already there are skipped,
so a preempted cluster job resumes where it stopped.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SELECTORS = ("search", "oracle", "sre", "sre_il", "heuristic", "planner", "nearest", "random", "clip", "gpt4o")


# ── ground truth from PyBullet ───────────────────────────────────────────────────────────────────────────────────────

def mask_bodies(masks, seg, object_ids, min_agree=0.3):
    """Body id behind each mask: the most common object id among the mask's pixels in the top camera's seg image."""
    out = []
    for m in masks:
        s = seg if seg.shape == m.shape else cv2.resize(seg, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        vals = s[m > 0]
        vals = vals[np.isin(vals, object_ids)]
        if vals.size == 0:
            out.append(None)
            continue
        ids, counts = np.unique(vals, return_counts=True)
        best = int(ids[np.argmax(counts)])
        out.append(best if counts.max() >= min_agree * np.count_nonzero(m) else None)
    return out


def touching_bodies(p, target_body, object_ids, dist):
    """Objects within `dist` metres of the target: the ones that can be in the way of grasping it."""
    return sorted(int(o) for o in object_ids
                  if o != target_body and p.getClosestPoints(bodyA=int(o), bodyB=int(target_body), distance=dist))


def top_z(p, body):
    return p.getAABB(int(body))[1][2]


# ── mask corruption (Item 3) ─────────────────────────────────────────────────────────────────────────────────────────

def corrupt_masks(masks, bboxes, spec, rng, protect=None):
    """Apply one corruption type at one level. `protect` = index kept intact (the target), or None."""
    if not spec:
        return masks, bboxes, []
    kind, level = spec.split(":")
    level = float(level)
    masks = [m.copy() for m in masks]
    log = []
    if kind in ("erode", "dilate"):
        k = int(level)
        ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        op = cv2.erode if kind == "erode" else cv2.dilate
        masks = [op(m, ker) for m in masks]
        log.append(f"{kind}{k}")
    elif kind == "drop":
        keep = [i for i in range(len(masks)) if i == protect or rng.rand() >= level]
        log += [f"drop{i}" for i in range(len(masks)) if i not in keep]
        masks = [masks[i] for i in keep]
    elif kind == "split":
        new = []
        for i, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            if i != protect and len(xs) > 200 and rng.rand() < level:
                pts = np.stack([xs, ys], 1).astype(np.float32)
                mean = pts.mean(0)
                axis = np.linalg.svd(pts - mean, full_matrices=False)[2][0]
                side = (pts - mean) @ axis > 0
                for s in (side, ~side):
                    h = np.zeros_like(m)
                    h[ys[s], xs[s]] = 255
                    new.append(h)
                log.append(f"split{i}")
            else:
                new.append(m)
        masks = new
    elif kind == "merge":
        ker = np.ones((7, 7), np.uint8)
        used, new = set(), []
        for i in range(len(masks)):
            if i in used:
                continue
            m = masks[i]
            if i != protect and rng.rand() < level:
                grown = cv2.dilate(m, ker)
                for j in range(i + 1, len(masks)):
                    if j not in used and j != protect and np.any(grown & masks[j]):
                        m = np.maximum(m, masks[j])
                        used.add(j)
                        log.append(f"merge{i}+{j}")
                        break
            new.append(m)
            used.add(i)
        masks = new
    else:
        raise ValueError(f"unknown mask noise '{kind}'")
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else [0, 0, 1, 1])
    return masks, boxes, log


# ── harness ──────────────────────────────────────────────────────────────────────────────────────────────────────────

class Harness:
    def __init__(self, args):
        import torch
        import yaml
        import pybullet as p

        from env.environment import Environment
        from mask_rg.object_segmenter import ObjectSegmenter
        from policy.ae_model import ActionDecoder
        from policy.sre_actor_critic import SREActorCritic
        from policy.sre_model import SpatialEncoder

        self.args, self.p, self.torch = args, p, torch
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        with open(args.config) as f:
            self.params = yaml.safe_load(f)
        self.rotations = self.params['agent']['fcn']['rotations']
        self.aperture_limits = self.params['agent']['regressor']['aperture_limits']
        self.pxl_size = self.params['env']['pixel_size']
        self.bounds = np.array(self.params['env']['workspace']['bounds'])
        self.push_distance, self.z = 0.10, 0.08   # as Policy

        self.env = Environment(self.params, objects_set=args.objects_set, render=args.render,
                               nr_objects=args.nr_objects)
        model_args = Namespace(device=self.device, num_patches=args.num_patches, sequence_length=1)
        self.segmenter = ObjectSegmenter(model_args)
        self.seg_dir = tempfile.mkdtemp(prefix="selector_seg_")

        self.ae = ActionDecoder(model_args).to(self.device)
        self.ae.load_state_dict(torch.load(args.ae_model, map_location=self.device))
        self.ae.eval()
        self.sre_il = self.sre_rl = None
        if "sre_il" in args.selectors:
            self.sre_il = SpatialEncoder(model_args).to(self.device)
            self.sre_il.load_state_dict(torch.load(args.sre_model, map_location=self.device))
            self.sre_il.eval()
        if "sre" in args.selectors:
            self.sre_rl = SREActorCritic(model_args).to(self.device)
            ckpt = torch.load(args.sre_rl, map_location=self.device, weights_only=False)
            self.sre_rl.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            self.sre_rl.eval()
        self._clip = self._gpt = None
        self.exit_args = None
        if args.execution == "ideal":       # the search's own defaults, so the test is the one the labels used
            from trainer.train_sre_exit import parse_args as exit_parse_args
            self.exit_args = exit_parse_args(["--access", args.access])
        elif "search" in args.selectors:
            raise SystemExit("the 'search' selector needs --execution ideal")

    # ── one episode ──

    def run_episode(self, episode, seed, selector):
        import policy.grasping as grasping
        import utils.general_utils as general_utils
        import env.cameras as cameras

        p, env = self.p, self.env
        rng = np.random.RandomState(seed)          # selector-side randomness, identical across selectors
        env.seed(seed)
        obs = env.reset()
        for _ in range(10):                        # same validity check as eval_agent (Policy.is_state_init_valid)
            if self._init_valid(obs):
                break
            obs = env.reset()

        object_ids = [o.body_id for o in env.objects]
        masks, bboxes = self._segment(obs)
        if not masks:
            return {"episode": episode, "seed": seed, "selector": selector, "skipped": "no objects segmented"}
        _, target_id = general_utils.get_target_mask(masks, obs['color'][1], rng)
        bodies = mask_bodies(masks, obs['seg'][1], object_ids)
        target_body = bodies[target_id]
        if target_body is None:
            return {"episode": episode, "seed": seed, "selector": selector, "skipped": "target mask has no body"}
        ref_target_mask = masks[target_id]
        search = None
        if self.exit_args is not None:
            from trainer.train_sre_exit import Search, make_probe
            search = Search(p, make_probe(p, self.exit_args), self.exit_args)    # env.reset cleared all bodies

        rec = {"episode": episode, "seed": seed, "selector": selector, "nr_objects": env.scene_nr_objs,
               "n_objects_initial": len(object_ids), "n_segmented_initial": len(masks),
               "target_touching_initial": len(touching_bodies(p, target_body, object_ids, self.args.contact_dist)),
               "target_visible_px_initial": int(np.count_nonzero(obs['seg'][1] == target_body)),
               "mask_noise": self.args.mask_noise, "steps": []}
        outcome, success = "step_limit", False

        for step in range(self.args.max_steps):
            object_ids = [o.body_id for o in env.objects]
            if step > 0:
                masks, bboxes = self._segment(obs)
                bodies = mask_bodies(masks, obs['seg'][1], object_ids)
            # target identity from ground truth; a hidden target keeps its last mask (as the real robot does)
            t_idx = [i for i, b in enumerate(bodies) if b == target_body]
            target_id = max(t_idx, key=lambda i: np.count_nonzero(masks[i])) if t_idx else -1
            target_mask = masks[target_id] if target_id >= 0 else ref_target_mask
            ref_target_mask = target_mask
            masks, bboxes, noise_log = corrupt_masks(masks, bboxes, self.args.mask_noise, rng,
                                                     protect=target_id if target_id >= 0 else None)
            bodies = mask_bodies(masks, obs['seg'][1], object_ids)
            t_idx = [i for i, b in enumerate(bodies) if b == target_body]
            target_id = max(t_idx, key=lambda i: np.count_nonzero(masks[i])) if t_idx else -1

            touching = touching_bodies(p, target_body, object_ids, self.args.contact_dist)
            valid = set(touching) if touching else {target_body}
            q = v0 = graspable = None
            if search is not None:
                root = p.saveState()
                search.reset(root, target_body, object_ids)
                q, v0, graspable = search.q_values(bodies)
                p.removeState(root)
                if step == 0:
                    if v0 > self.exit_args.max_depth + 1:
                        return {"episode": episode, "seed": seed, "selector": selector,
                                "skipped": "no solution within the search horizon"}
                    rec["optimal_steps"] = int(v0)
            ctx = dict(obs=obs, masks=masks, bboxes=bboxes, target_mask=target_mask, target_id=target_id,
                       bodies=bodies, valid=valid, rng=rng, q=q)

            t0 = time.perf_counter()
            chosen, probs = self._select(selector, ctx)
            sel_ms = 1e3 * (time.perf_counter() - t0)
            heur = self._select_heuristic(ctx)
            n = len(masks)
            if chosen is not None and chosen >= n:          # SRE / GPT convention: past the list = the target
                chosen = target_id
            chosen_body = bodies[chosen] if chosen is not None and 0 <= chosen < n else None
            st = {"step": step, "n_masks": n, "target_index": target_id, "chosen": chosen,
                  "chosen_body": chosen_body, "target_body": int(target_body), "chosen_is_target":
                  chosen_body == target_body, "valid_bodies": sorted(int(v) for v in valid),
                  "sel_valid": chosen_body in valid, "heuristic_choice": heur,
                  "agrees_heuristic": chosen == heur, "probs": probs, "noise": noise_log,
                  "select_ms": round(sel_ms, 1)}
            if search is not None:
                st.update(q=[float(v) for v in q], v0=int(v0), graspable=bool(graspable),
                          sel_optimal=chosen is not None and 0 <= chosen < n and bool(q[chosen] <= q.min()))

            if chosen is None or chosen < 0 or chosen >= n:
                st.update(exec_skipped=True, exec_ok=False, removed=[], stable=None)
                rec["steps"].append(st)
                continue

            if search is not None:
                # ideal execution: a graspable target is retrieved, a blocked one costs the step; any other object
                # is lifted out and the pile settles (a toppled or displaced target ends the episode)
                if chosen_body == target_body and graspable:
                    st.update(removed=[int(target_body)], exec_ok=True)
                    rec["steps"].append(st)
                    success, outcome, rec["success_intended"] = True, "success", True
                    break
                if chosen_body is None or chosen_body == target_body:
                    st.update(removed=[], exec_ok=False)
                    rec["steps"].append(st)
                    continue
                p.resetBasePositionAndOrientation(chosen_body, [20.0, 20.0, -0.6], [0, 0, 0, 1])
                for _ in range(self.exit_args.settle_steps):
                    p.stepSimulation()
                target_ok = search._target_ok()
                env.remove_flat_objs()
                obs = env.get_observation()
                st.update(removed=[int(chosen_body)], exec_ok=True)
                rec["steps"].append(st)
                if not target_ok or target_body not in [o.body_id for o in env.objects]:
                    outcome = "target_disturbed"
                    break
                continue

            before = set(object_ids)
            action = self._action_decoder(obs, masks[chosen])
            _, info = env.step(self._action3d(action))
            for _ in range(240):                         # let the dropped object land off the table (1 s)
                p.stepSimulation()
            env.remove_flat_objs()
            obs = env.get_observation()
            after = set(o.body_id for o in env.objects)
            removed = sorted(int(b) for b in before - after)
            st.update(removed=removed, stable=bool(info['stable']), collision=bool(info['collision']),
                      exec_ok=bool(info['stable']) and chosen_body in removed)
            rec["steps"].append(st)

            if target_body in removed:
                # the target left the table: in the hand (stable grasp, nothing else removed) or knocked over / off
                # (Environment.remove_flat_objs deletes toppled objects). A grasp aimed at an obstacle that lifted
                # the target still counts, as it did in the human-labelled eval; success_intended tells them apart.
                success = bool(info['stable']) and removed == [int(target_body)]
                rec["success_intended"] = success and chosen_body == target_body
                outcome = ("success" if success else
                           "target_toppled" if not info['stable'] else "target_removed_with_others")
                break
            if len(after) == 0:
                outcome = "scene_empty"
                break

        rec.setdefault("success_intended", False)
        rec.update(success=success, outcome=outcome, n_steps=len(rec["steps"]))
        return rec

    # ── pieces ──

    def _init_valid(self, obs):
        from utils.orientation import Quaternion
        flat = 0
        for obj in obs['full_state']:
            _, q = self.p.getBasePositionAndOrientation(obj.body_id)
            rz = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]).rotation_matrix()[0:3, 2]
            flat += np.abs(np.arccos(np.dot([0, 0, 1], rz))) > 0.1
        return flat != len(obs['full_state'])

    def _segment(self, obs):
        masks, _, _, bboxes = self.segmenter.from_maskrcnn(obs['color'][1], dir=self.seg_dir, bbox=True)
        return masks, bboxes

    def _action_decoder(self, obs, obstacle_mask):
        """Same as RLEnvironmentWrapper.get_action_from_policy: AD grasp on the chosen mask, mid aperture."""
        import utils.general_utils as gu
        import env.cameras as cameras
        torch = self.torch
        state = gu.get_fused_heightmap(obs, cameras.RealSense.CONFIG, self.bounds, self.pxl_size)
        obstacle = torch.FloatTensor(gu.preprocess_target(obstacle_mask, state)).unsqueeze(0).to(self.device)
        heightmap, pad = gu.preprocess_image(state)
        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = gu.postprocess(self.ae(x, obstacle, is_volatile=True), pad)
        best = np.unravel_index(np.argmax(out), out.shape)
        return np.array([best[3], best[2], best[0] * 2 * np.pi / self.rotations,
                         (self.aperture_limits[0] + self.aperture_limits[1]) / 2])

    def _action3d(self, action):
        import utils.orientation as ori
        x = -(self.pxl_size * action[0] - self.bounds[0][1])
        y = self.pxl_size * action[1] - self.bounds[1][1]
        quat = ori.Quaternion.from_rotation_matrix(np.matmul(ori.rot_y(-np.pi / 2), ori.rot_x(action[2])))
        return {'pos': np.array([x, y, self.z]), 'quat': quat, 'aperture': action[3],
                'push_distance': self.push_distance}

    # ── selectors ──

    def _select(self, name, c):
        n, t = len(c["masks"]), c["target_id"]
        if name == "random":
            return int(c["rng"].randint(n)), None
        if name == "heuristic":
            return self._select_heuristic(c), None
        if name == "nearest":
            return self._select_nearest(c), None
        if name == "planner":
            return self._select_planner(c), None
        if name == "oracle":
            return self._select_oracle(c), None
        if name == "search":
            return int(np.argmin(c["q"])), None
        if name in ("sre", "sre_il"):
            return self._select_sre(name, c)
        if name == "clip":
            if self._clip is None:
                from baseline.clip_eval import ZeroShotCLIPRemovalPredictor
                self._clip = ZeroShotCLIPRemovalPredictor()
            cand = [i for i in range(n) if i != t] or [t]
            sub = [c["masks"][i].astype(np.float32) / 255.0 for i in cand]
            return cand[self._clip.predict_removal(sub, c["target_mask"].astype(np.float32) / 255.0)], None
        if name == "gpt4o":
            if self._gpt is None:
                from baseline.gpt import GPTRemovalPredictor
                self._gpt = GPTRemovalPredictor()
            return int(self._gpt.predict(list(c["masks"]), c["target_mask"])), None
        raise ValueError(name)

    def _select_heuristic(self, c):
        """Algorithm 1 exactly as eval_agent.run_episode_heuristics uses it."""
        import policy.grasping as grasping
        masks, t = c["masks"], c["target_id"]
        if t < 0:
            pool, t = list(masks) + [c["target_mask"]], len(masks)
        else:
            pool = masks
        try:
            order = grasping.find_obstacles_to_remove(t, pool)
        except (ValueError, ZeroDivisionError, IndexError):
            order = [t]
        if len(order) < 4 and t in order:
            order.remove(t)
            order = [t] + order
        return order[0]

    def _select_nearest(self, c):
        masks, t = c["masks"], c["target_id"]
        tc = _centroid(c["target_mask"])
        others = [i for i in range(len(masks)) if i != t]
        if not others:
            return t
        return min(others, key=lambda i: np.hypot(*np.subtract(_centroid(masks[i]), tc)))

    def _select_planner(self, c, contact_px=6):
        """Classical occlusion-graph planner from perception only (masks + top-camera depth).

        Blockers of X = objects whose mask touches X's mask (within contact_px). Among the target's blockers take the
        one whose top is highest (smallest camera depth); if something higher rests on that blocker, clear it first.
        No blockers = grasp the target."""
        masks, t = c["masks"], c["target_id"]
        depth = c["obs"]['depth'][1]
        ker = np.ones((2 * contact_px + 1, 2 * contact_px + 1), np.uint8)

        def top(i):
            m = masks[i] if depth.shape == masks[i].shape else cv2.resize(masks[i], depth.shape[::-1],
                                                                            interpolation=cv2.INTER_NEAREST)
            d = depth[m > 0]
            return float(np.percentile(d, 5)) if d.size else np.inf

        def blockers(ref_mask, exclude):
            g = cv2.dilate(ref_mask, ker)
            return [i for i in range(len(masks)) if i not in exclude and np.any(g & masks[i])]

        target_mask = c["target_mask"]
        exclude = {t} if t >= 0 else set()
        bl = blockers(target_mask, exclude)
        if not bl:
            return t if t >= 0 else self._select_nearest(c)
        chosen = min(bl, key=top)
        for _ in range(len(masks)):                       # climb to whatever sits higher on the chosen blocker
            above = [i for i in blockers(masks[chosen], exclude | {chosen}) if top(i) < top(chosen) - 0.005]
            if not above:
                break
            chosen = min(above, key=top)
        return chosen

    def _select_oracle(self, c):
        """Ground truth: the highest object that touches the target; the target once nothing touches it."""
        bodies, valid = c["bodies"], c["valid"]
        cands = [i for i, b in enumerate(bodies) if b in valid]
        if not cands:
            return c["target_id"] if c["target_id"] >= 0 else self._select_nearest(c)
        return max(cands, key=lambda i: top_z(self.p, bodies[i]))

    def _select_sre(self, name, c):
        import utils.general_utils as gu
        torch = self.torch
        masks, bboxes = c["masks"], c["bboxes"]
        k = min(len(masks), self.args.num_patches)
        scene = torch.FloatTensor(gu.resize_mask(c["obs"]['color'][1]).mean(axis=2)).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(gu.resize_mask(c["target_mask"])).unsqueeze(0).to(self.device)
        m_t = torch.zeros((1, self.args.num_patches, 1, 100, 100), device=self.device)
        b_t = torch.zeros((1, self.args.num_patches, 4), device=self.device)
        for i in range(k):
            m_t[0, i, 0] = torch.FloatTensor(gu.resize_mask(masks[i]))
            b_t[0, i] = torch.FloatTensor(gu.resize_bbox(bboxes[i]))
        with torch.no_grad():
            logits = (self.sre_rl(scene, target, m_t, b_t)[0] if name == "sre" else
                      self.sre_il(scene, target, m_t, b_t)[0])[0].float()
        probs = [round(v, 4) for v in torch.softmax(logits[:k], 0).tolist()]
        return int(torch.argmax(logits).item()), probs


def _centroid(mask):
    m = cv2.moments((np.asarray(mask) > 0).astype(np.uint8))
    return [0.0, 0.0] if m["m00"] == 0 else [m["m10"] / m["m00"], m["m01"] / m["m00"]]


# ── summary ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def summarize(out_dir):
    rows = [json.loads(l) for l in open(Path(out_dir) / "episodes.jsonl") if l.strip()]
    rows = [r for r in rows if "skipped" not in r]
    by = defaultdict(list)
    for r in rows:
        by[r["selector"]].append(r)
    # episodes every selector finished, so each row compares like with like
    common = set.intersection(*(set(r["episode"] for r in v) for v in by.values())) if by else set()
    print(f"{len(common)} episodes completed by all {len(by)} selectors ({out_dir})\n")
    hdr = f"{'selector':<10} {'success':>10} {'rate':>6} {'steps|succ':>10} {'eps_SRE':>8} {'eps_exec':>8} {'agree_H':>8}  outcomes"
    print(hdr)
    print("-" * len(hdr))
    order = [s for s in SELECTORS if s in by] + [s for s in by if s not in SELECTORS]
    for s in order:
        eps = [r for r in by[s] if r["episode"] in common]
        if not eps:
            continue
        succ = [r for r in eps if r["success"]]
        n_intended = sum(r.get("success_intended", False) for r in eps)
        steps = [st for r in eps for st in r["steps"]]
        executed = [st for st in steps if not st.get("exec_skipped")]
        valid_exec = [st for st in executed if st["sel_valid"]]
        e_sre = 1 - np.mean([st["sel_valid"] for st in steps]) if steps else float("nan")
        e_exec = 1 - np.mean([st["exec_ok"] for st in valid_exec]) if valid_exec else float("nan")
        agree = np.mean([st["agrees_heuristic"] for st in steps]) if steps else float("nan")
        outc = defaultdict(int)
        for r in eps:
            outc[r["outcome"]] += 1
        mean_steps = np.mean([r["n_steps"] for r in succ]) if succ else float("nan")
        print(f"{s:<10} {len(succ):>4}/{len(eps):<5} {len(succ) / len(eps):>6.1%} {mean_steps:>10.2f} "
              f"{e_sre:>8.3f} {e_exec:>8.3f} {agree:>8.1%}  intended={n_intended} {dict(outc)}")
    print("\neps_SRE = share of steps choosing an object that does not touch the target (or not the free target);"
          "\neps_exec = share of valid choices the grasp failed to remove; agree_H = step choices equal to Alg. 1.")
    if any("optimal_steps" in r for r in rows):
        summarize_ideal(by, common, order)


def summarize_ideal(by, common, order):
    """Selection-only columns for --execution ideal runs, with the search's cost-to-go as the reference."""
    rng = np.random.RandomState(0)
    print(f"\nideal execution (selection only)\n{'selector':<10} {'success':>10} {'95% interval':>14} "
          f"{'needs removal':>14} {'free target':>12} {'opt choice':>11} {'excess steps':>13} {'early grasp':>12}")
    for s in order:
        eps = [r for r in by[s] if r["episode"] in common]
        if not eps:
            continue
        ok = np.array([r["success"] for r in eps])
        lo, hi = np.percentile([rng.choice(ok, len(ok)).mean() for _ in range(2000)], [2.5, 97.5])
        hard = [r["success"] for r in eps if r["optimal_steps"] > 1]
        easy = [r["success"] for r in eps if r["optimal_steps"] == 1]
        steps = [st for r in eps for st in r["steps"]]
        excess = [r["n_steps"] - r["optimal_steps"] for r in eps if r["success"]]
        early = sum(any(st["chosen_is_target"] and not st["graspable"] for st in r["steps"]) for r in eps)
        print(f"{s:<10} {ok.sum():>4}/{len(ok):<5} {f'[{lo:.0%}, {hi:.0%}]':>14} {f'{sum(hard)}/{len(hard)}':>14} "
              f"{f'{sum(easy)}/{len(easy)}':>12} {np.mean([st['sel_optimal'] for st in steps]):>11.1%} "
              f"{np.mean(excess) if excess else float('nan'):>13.2f} {f'{early}/{len(eps)}':>12}")
    print("\nneeds removal / free target = scenes where the search's optimum is more than one action / one action;"
          "\nopt choice = share of steps whose choice has the minimum search cost; excess steps = steps beyond the"
          "\noptimum on successful episodes; early grasp = episodes with a grasp at the target while it was still blocked"
          "\n(the step is wasted and the scene unchanged, so a deterministic selector repeats it to the step limit).")


# ── main ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", metavar="OUT_DIR", default=None)
    ap.add_argument("--selectors", nargs="+", default=["oracle", "sre", "sre_il", "heuristic", "planner", "nearest",
                                                       "random"], choices=SELECTORS)
    ap.add_argument("--n_scenes", type=int, default=30)
    ap.add_argument("--nr_objects", type=int, nargs=2, default=[6, 9], metavar=("LOW", "HIGH"),
                    help="objects per scene, [low, high) as Environment.nr_objects")
    ap.add_argument("--seed", type=int, default=1, help="scene seeds are drawn from this, as in eval_agent")
    ap.add_argument("--max_steps", type=int, default=6, help="eval_agent uses 6")
    ap.add_argument("--contact_dist", type=float, default=0.01, help="metres; 'touching the target' for ground truth")
    ap.add_argument("--execution", default="ad", choices=["ad", "ideal"],
                    help="ad: Action Decoder grasps in physics; ideal: lift the chosen object out (selection only)")
    ap.add_argument("--access", default="side", choices=["side", "top"],
                    help="--execution ideal: graspability test of trainer/train_sre_exit.py")
    ap.add_argument("--mask_noise", default=None, help="e.g. merge:0.3, split:0.3, drop:0.2, erode:4, dilate:4")
    ap.add_argument("--render", default="egl", choices=["gui", "direct", "egl"])
    ap.add_argument("--objects_set", default="unseen", help="eval_agent uses 'unseen'")
    ap.add_argument("--out", default="save/selector_eval/run")
    ap.add_argument("--config", default="yaml/bhand.yml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_patches", type=int, default=10)
    ap.add_argument("--ae_model", default="save/ae/ae_model_best.pt")
    ap.add_argument("--sre_model", default="save/sre/sre_model_best.pt")
    ap.add_argument("--sre_rl", default="save/sre_rl/sre_rl_best.pt")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "episodes.jsonl"
    done = set()
    if log_path.exists():
        for l in open(log_path):
            if l.strip():
                r = json.loads(l)
                done.add((r["episode"], r["selector"]))
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    harness = Harness(args)
    seed_rng = np.random.RandomState(args.seed)
    seeds = [int(seed_rng.randint(0, 2 ** 32 - 1)) for _ in range(args.n_scenes)]
    for ep, seed in enumerate(seeds):
        for sel in args.selectors:
            if (ep, sel) in done:
                continue
            t0 = time.time()
            rec = harness.run_episode(ep, seed, sel)
            rec["duration_s"] = round(time.time() - t0, 1)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, default=int) + "\n")
            print(f"ep {ep:3d} {sel:<9} {rec.get('outcome', rec.get('skipped'))} "
                  f"steps={rec.get('n_steps', '-')} ({rec['duration_s']} s)", flush=True)
    summarize(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
