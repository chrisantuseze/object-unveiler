"""Fixed scenes for the real-robot evaluation (Task 1 of docs/sre_training_and_real_eval.md), built in the twin, so
every method is run on the same arrangements (as verify2act/twin/make_eval_layouts.py does for Verify2Act).
Task 2 scenes (two targets, retrieved in order) follow the Task 1 ones: T2a puts each target under its own block,
T2b lays one block across both targets.

Three kinds of scene, each with a target block:
  C1  one block lies on the target        (the search needs 1 removal, then the target: 2 steps)
  C2  two blocks lie on the target        (2 removals: 3 steps)
  F   the target is free (control)        (1 step)
A layout is kept only if
  * every block is inside the workspace view of the arm camera and the target did not move when it was covered,
  * the search (twin/unveil.py, the top-grasp test) gives exactly the step count of its kind,
  * that count and the set of best first choices stay the same when the scene is rebuilt --robust times with
    placement noise (--noise_pos m, --noise_yaw deg): a hand-built copy has the same answer.
No model is run on the layouts, so the choice of scenes cannot favour a method.

Per layout the sheet shows a to-scale top view (block centres in cm from the far edge and the left edge of the
Letter-size workspace rectangle, as seen from the robot), the build order, and the expected arm-camera view before
and after the covering blocks go on.

    python -m twin.make_eval_layouts --out twin/real_eval_layouts
    python -m twin.make_eval_layouts --check C1-03 --frame <frame.jpg>     # a real frame against the layout
    python -m twin.make_eval_layouts --check C1-03 --frame save/real_eval/<session>     # ... its newest frame.jpg
    python -m twin.make_eval_layouts --check grid --frame <frame.jpg>      # only the workspace rectangle, 5 cm grid
"""

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from twin.unveil import SIM_SIZE, UnveilTwin, parse_args as unveil_args   # noqa: E402

KINDS = {"C1": (1, 2, "one block on the target"), "C2": (2, 3, "two blocks on the target"),
         "F": (0, 1, "free target (control)")}
# Task 2, two targets in order: (removals before the first target, before the second, text)
KINDS2 = {"T2a": (1, 1, "each target under its own block"), "T2b": (1, 0, "one block lies across both targets")}
RGB = {"red": (185, 35, 30), "green": (30, 95, 60), "blue": (35, 105, 200), "yellow": (238, 205, 45)}
PX_PER_CM = 22
MIN_TARGET_PX = 150        # twin/eval.py and the Jetson runner: a smaller visible target counts as hidden


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


# ── building a scene ──

def construct(ut, table, drops, rng, noise_pos=0.0, noise_yaw=0.0):
    """Table blocks at their poses, then the covering blocks released one by one above the target."""
    tw = ut.tw
    covers = [d[0] for d in drops]
    n = lambda s: float(rng.normal(0, s)) if s else 0.0
    state = {}
    for c in tw.colors:
        s = table[c]
        on_table = s["present"] and c not in covers
        state[c] = {"present": on_table, "yaw": s["yaw"] + n(noise_yaw),
                    "pos": [s["pos"][0] + n(noise_pos), s["pos"][1] + n(noise_pos), s["pos"][2]]}
    tw.set_state(state)
    tw.settle()
    for c, x, y, yaw in drops:
        tw.present[c] = True
        tw._drop(c, x + n(noise_pos), y + n(noise_pos), yaw + n(noise_yaw), rng)


def show_only(ut, state, keep):
    """The saved scene with only the blocks in ``keep`` (the others parked out of view)."""
    ut.restore(state)
    for c in ut.present():
        if c not in keep:
            ut.remove(c, settle=False)


def solve(ut, target):
    """(steps of the search's optimum, best first choices, one optimal plan)."""
    ut.start_search(target)
    root = ut.root
    v0 = ut.value(())
    plan, removed = [], ()
    first = None
    while True:
        ok, graspable, cands = ut._state(removed)
        if not ok or graspable or not cands or len(removed) >= ut.a.max_depth:
            best = [target] if ok and graspable else []
        else:
            cost = {c: ut.value(removed + (c,)) for c in cands}
            best = sorted(c for c in cands if cost[c] == min(cost.values()))
        if first is None:
            first = best
        if not best:
            break
        top = max(best, key=lambda c: ut.frame(c)[0][2])      # of equally good blocks, the one on top
        plan.append(top)
        if top == target:
            break
        removed += (top,)
    ut.restore(root)
    return int(v0), first, plan


def sample(ut, rng, kind, target, others):
    """One candidate layout of this kind, or None. The first blocks of ``others`` go on the target."""
    a, tw = ut.a, ut.tw
    k = KINDS[kind][0]
    tw.episode_camera = dict(tw.base_camera)
    try:
        tw.reset(rng, [target] + others)
    except RuntimeError:
        return None
    table = tw.get_state()
    p, yaw = tw.pose(target)
    cs, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    drops = []
    for c in others[:k]:
        d = rng.uniform(-a.cover_offset, a.cover_offset, 2) * ut.half[:2]
        drops.append((c, float(p[0] + cs * d[0] - sn * d[1]), float(p[1] + sn * d[0] + cs * d[1]),
                      float(yaw + rng.uniform(0, 180))))
    return {"kind": kind, "target": target, "blocks": [target] + others, "table": table, "drops": drops,
            "seed": int(rng.integers(1 << 31))}


def sample2(ut, rng, kind, a_t, b_t, others):
    """Task 2 candidate: targets ``a_t`` then ``b_t``, both covered. T2a: others[0] on a_t and others[1] on b_t.
    T2b: b_t lies beside a_t and others[0] across both; others[1] is a bystander."""
    a, tw = ut.a, ut.tw
    tw.episode_camera = dict(tw.base_camera)
    try:
        tw.reset(rng, [a_t, b_t] + others)
    except RuntimeError:
        return None
    p, yaw = tw.pose(a_t)
    cs, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    over = lambda q, y: (float(q[0] + math.cos(math.radians(y)) * d[0] - math.sin(math.radians(y)) * d[1]),
                         float(q[1] + math.sin(math.radians(y)) * d[0] + math.cos(math.radians(y)) * d[1]))
    if kind == "T2a":
        drops = []
        for c, t in zip(others, (a_t, b_t)):
            q, y = tw.pose(t)
            d = rng.uniform(-a.cover_offset, a.cover_offset, 2) * ut.half[:2]
            drops.append((c, *over(q, y), float(y + rng.uniform(0, 180))))
    else:
        side = rng.choice([-1.0, 1.0]) * (2 * ut.half[1] + rng.uniform(0.016, 0.022))    # room for a finger between
        x, y, yb = p[0] - sn * side, p[1] + cs * side, yaw + rng.normal(0, 4)
        for c in (b_t,):                                # move b_t next to a_t, if the spot is free
            tw.present[c] = False
            if not tw._free_spot(c, x, y, yb):
                return None
            tw.present[c] = True
            tw._set_pose(c, (x, y, p[2]), yb)
        ut.mj.mj_forward(tw.model, tw.data)
        tw.settle()
        d = np.array([rng.uniform(-0.4, 0.4) * ut.half[0], 0.0])
        mid = (np.asarray(p[:2]) + np.array([x, y])) / 2
        drops = [(others[0], *over(mid, yaw), float(yaw + 90 + rng.uniform(-15, 15)))]
    return {"kind": kind, "target": a_t, "targets": [a_t, b_t], "blocks": [a_t, b_t] + others,
            "table": tw.get_state(), "drops": drops, "seed": int(rng.integers(1 << 31))}


def phase(ut, target, forbidden=()):
    """Fewest removals, none of them in ``forbidden``, after which the target can be grasped from the top:
    (count, blocks that are in a smallest set, one set with the highest block first), or None."""
    ut.start_search(target)
    root = ut.root
    allowed = [c for c in ut.present() if c != target and c not in forbidden]
    height = {c: ut.frame(c)[0][2] for c in allowed}
    out = None
    for n in range(len(allowed) + 1):
        good = [seq for seq in itertools.combinations(allowed, n) if all(ut._state(seq)[:2])]
        if good:
            out = (n, sorted({c for seq in good for c in seq}), sorted(good[0], key=lambda c: -height[c]))
            break
    ut.restore(root)
    return out


def solve2(ut, targets):
    """The ordered task: (removals before each target, best first choices, the whole plan), or None."""
    start = ut.save()
    counts, plan, first = [], [], None
    for k, t in enumerate(targets):
        r = phase(ut, t, forbidden=targets[k + 1:])
        if r is None or not ut.tw.present[t]:
            ut.restore(start)
            return None
        counts.append(r[0])
        if first is None:
            first = r[1] or [t]
        for c in r[2] + [t]:
            ut.remove(c, settle=False)
        ut.tw.settle(ut.a.settle_steps)
        plan += r[2] + [t]
    ut.restore(start)
    return counts, first, plan


def geometry_ok(ut, lay, args, targets):
    """Every block in the workspace view and not on its end; the targets where the reference photo saw them."""
    tw, blocks = ut.tw, lay["blocks"]
    if not ut._in_workspace(blocks) or not all(tw.in_view(*tw.pose(c), margin=args.margin) for c in blocks):
        return False
    uv = tw.project(np.array([tw.pose(c)[0] for c in blocks]))
    w = cv2.perspectiveTransform(uv.reshape(-1, 1, 2).astype(np.float32), ut.H).reshape(-1, 2)
    if not ((w > args.view_margin) & (w < SIM_SIZE - args.view_margin)).all():
        return False
    for t in targets:
        p, _ = tw.pose(t)
        if np.linalg.norm(p[:2] - np.array(lay["table"][t]["pos"][:2])) > 0.004 or tw.is_tipped(t, 4.0):
            return False
    return not any(max(tilt(ut.frame(c)[1])) > args.max_tilt for c in blocks)


def seen_share(ut, state, target):
    """(pixels of the target the arm camera sees now, pixels when it is alone on the table)."""
    tw, t_i = ut.tw, ut.tw.colors.index(target)
    ut.restore(state)
    seen = int((ut.block_ids(tw.base_camera) == t_i).sum())
    show_only(ut, state, [target])
    alone = int((ut.block_ids(tw.base_camera) == t_i).sum())
    ut.restore(state)
    return seen, alone


def accept2(ut, lay, args):
    """As ``accept``, for the ordered two-target task."""
    targets = lay["targets"]
    construct(ut, lay["table"], lay["drops"], np.random.default_rng(lay["seed"]))
    if not geometry_ok(ut, lay, args, targets) or any(ut.graspable(t) for t in targets):
        return False                                   # both targets start covered
    state = ut.save()
    sol = solve2(ut, targets)
    if sol is None or tuple(sol[0]) != KINDS2[lay["kind"]][:2]:
        return False
    rng = np.random.default_rng(lay["seed"] + 1)
    for _ in range(args.robust):
        construct(ut, lay["table"], lay["drops"], rng, args.noise_pos, args.noise_yaw)
        if any(ut.graspable(t) for t in targets):
            return False
        s = solve2(ut, targets)
        if s is None or s[0] != sol[0] or s[1] != sol[1]:
            return False
    px = [seen_share(ut, state, t) for t in targets]
    lay.update(optimal_steps=sum(sol[0]) + len(targets), first_choices=sol[1], plan=sol[2],
               target_px=[p[0] for p in px], target_px_alone=[p[1] for p in px],
               occlusion="/".join("full" if p[0] < MIN_TARGET_PX else "partial" for p in px),
               qpos=[round(float(v), 6) for v in state[0]], present={c: bool(v) for c, v in state[1].items()})
    return True


def tilt(R):
    """(pitch of the long side, roll about it), degrees."""
    pitch = math.degrees(math.asin(min(1.0, abs(R[2, 0]))))
    roll = math.degrees(math.acos(min(1.0, max(abs(R[2, 1]), abs(R[2, 2])) / max(math.cos(math.radians(pitch)), 1e-6))))
    return pitch, roll


def accept(ut, lay, args):
    """Build the layout; True when it is a clean, robust scene of its kind. Fills in the solution and the state."""
    tw, target = ut.tw, lay["target"]
    construct(ut, lay["table"], lay["drops"], np.random.default_rng(lay["seed"]))
    blocks = lay["blocks"]
    if not ut._in_workspace(blocks) or not all(tw.in_view(*tw.pose(c), margin=args.margin) for c in blocks):
        return False
    uv = tw.project(np.array([tw.pose(c)[0] for c in blocks]))
    w = cv2.perspectiveTransform(uv.reshape(-1, 1, 2).astype(np.float32), ut.H).reshape(-1, 2)
    if not ((w > args.view_margin) & (w < SIM_SIZE - args.view_margin)).all():
        return False
    p, _ = tw.pose(target)
    if np.linalg.norm(p[:2] - np.array(lay["table"][target]["pos"][:2])) > 0.004 or tw.is_tipped(target, 4.0):
        return False                                   # the target must stay where the reference photo saw it
    if any(max(tilt(ut.frame(c)[1])) > args.max_tilt for c in blocks):
        return False                                   # a block on its end or edge: hard to place by hand
    state = ut.save()
    v0, first, plan = solve(ut, target)
    if v0 != KINDS[lay["kind"]][1]:
        return False
    rng = np.random.default_rng(lay["seed"] + 1)
    for _ in range(args.robust):
        construct(ut, lay["table"], lay["drops"], rng, args.noise_pos, args.noise_yaw)
        v, f, _ = solve(ut, target)
        if v != v0 or f != first:
            return False
    ut.restore(state)
    ids = ut.block_ids(tw.base_camera)
    t_i = tw.colors.index(target)
    seen = int((ids == t_i).sum())
    show_only(ut, state, [target])
    alone = int((ut.block_ids(tw.base_camera) == t_i).sum())    # what the reference photo sees
    ut.restore(state)
    lay.update(optimal_steps=v0, first_choices=first, plan=plan, target_px=seen, target_px_alone=alone,
               occlusion="free" if lay["kind"] == "F" else "full" if seen < MIN_TARGET_PX else "partial",
               qpos=[round(float(v), 6) for v in state[0]], present={c: bool(v) for c, v in state[1].items()})
    return True


# ── the sheet ──

def supports(ut, c):
    """What block ``c`` rests on: lower blocks, and the table."""
    tw, mj = ut.tw, ut.mj
    names = {tw._geom[o]: o for o in tw.colors}
    table = tw.model.geom("table").id
    out = set()
    for i in range(tw.data.ncon):
        con = tw.data.contact[i]
        pair = (int(con.geom1), int(con.geom2))
        if tw._geom[c] not in pair:
            continue
        other = pair[0] if pair[1] == tw._geom[c] else pair[1]
        if other == table:
            out.add("the table")
        elif other in names and tw.present[names[other]] and ut.frame(names[other])[0][2] < ut.frame(c)[0][2]:
            out.add(names[other])                       # only what lies below it
    return out


def compass(v):
    """A horizontal direction in the operator's words (x = away from the robot, y = the robot's left)."""
    a = math.degrees(math.atan2(v[1], v[0])) % 360
    return ["far", "far-left", "left", "near-left", "near", "near-right", "right", "far-right"][int((a + 22.5) // 45) % 8]


def describe(ut, lay):
    """Build order: the target, the other table blocks, then the covering blocks. One line per block."""
    tw = ut.tw
    sx, sy = tw.cfg["sheet"]["size"]
    covers = [d[0] for d in lay["drops"]]
    targets = lay.get("targets", [lay["target"]])
    order = targets + [c for c in lay["blocks"] if c not in covers and c not in targets] + covers
    rows = []
    for i, c in enumerate(order):
        p, R = ut.frame(c)
        long = R[:, 0] if R[0, 0] >= 0 else -R[:, 0]
        rot = float(round(math.degrees(math.atan2(long[1], long[0])))) + 0.0
        pitch, roll = tilt(R)
        where = f"far edge {100 * (sx / 2 - p[0]):4.1f} cm   left edge {100 * (sy / 2 - p[1]):4.1f} cm   " \
                f"turned {rot + 0.0:+4.0f}° ({'far end to the left' if rot > 2 else 'far end to the right' if rot < -2 else 'straight'})"
        sup = supports(ut, c)
        if c in covers:
            on = sorted(sup - {"the table"})
            how = "lies on " + " and ".join(on) if on else "slid off: lies on the table"
            if "the table" in sup and on:
                how = "leans on " + " and ".join(on) + ", one end on the table"
            if pitch > 8:
                up = R[:, 0] * (1 if R[2, 0] > 0 else -1)
                how += f"; {compass(up)} end raised ({pitch:.0f}°)"
            elif roll > 8:
                side = max((R[:, 1], -R[:, 1], R[:, 2], -R[:, 2]), key=lambda v: v[2] - abs(v[2]) * 2 + np.hypot(*v[:2]))
                how += f"; tipped {roll:.0f}° toward {compass(-side)}"
            else:
                how += ", level"
        else:
            how = "flat on the table" + ("" if c not in targets else "  (TARGET)" if len(targets) == 1
                                         else f"  (TARGET {targets.index(c) + 1})")
        rows.append({"n": i + 1, "block": c, "text": f"{i + 1}. {c:6s} {where}   {how}",
                     "far_cm": round(100 * (sx / 2 - p[0]), 1), "left_cm": round(100 * (sy / 2 - p[1]), 1),
                     "turn_deg": round(rot), "cover": c in covers})
    return rows


def top_view(ut, lay, rows):
    """To-scale view from straight above: away from the robot is up, the robot's left is left."""
    tw = ut.tw
    sx, sy = tw.cfg["sheet"]["size"]
    W, H = int(sy * 100 * PX_PER_CM), int(sx * 100 * PX_PER_CM)
    pad_t, pad_b, pad_l, pad_r = 70, 60, 95, 30
    img = Image.new("RGB", (W + pad_l + pad_r, H + pad_t + pad_b), "white")
    d = ImageDraw.Draw(img)
    f, fs = font(18), font(14)
    to_px = lambda x, y: (pad_l + (sy / 2 - y) * 100 * PX_PER_CM, pad_t + (sx / 2 - x) * 100 * PX_PER_CM)
    d.rectangle([pad_l, pad_t, pad_l + W, pad_t + H], fill=(226, 214, 200), outline="black", width=2)
    for cm in range(1, int(sy * 100) + 1):
        x = pad_l + cm * PX_PER_CM
        d.line([x, pad_t, x, pad_t + H], fill=(205, 193, 180) if cm % 5 else (150, 140, 130))
        if cm % 5 == 0:
            d.text((x - 8, pad_t - 22), str(cm), fill="black", font=fs)
    for cm in range(1, int(sx * 100) + 1):
        y = pad_t + cm * PX_PER_CM
        d.line([pad_l, y, pad_l + W, y], fill=(205, 193, 180) if cm % 5 else (150, 140, 130))
        if cm % 5 == 0:
            d.text((pad_l - 30, y - 8), str(cm), fill="black", font=fs)
    d.text((pad_l, pad_t - 44), "cm from the LEFT edge  →", fill=(90, 90, 90), font=fs)
    d.text((4, pad_t + H // 2 - 40), "cm\nfrom\nFAR\nedge\n↓", fill=(90, 90, 90), font=fs)
    d.text((pad_l + W // 2 - 60, pad_t + H + 12), "▲ ROBOT", fill="black", font=f)
    num = {r["block"]: r["n"] for r in rows}
    g = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    for c in sorted(lay["blocks"], key=lambda c: ut.frame(c)[0][2]):
        p, R = ut.frame(c)
        pts = p + (g * ut.half) @ R.T
        hull = cv2.convexHull(np.float32([to_px(x, y) for x, y, _ in pts])).reshape(-1, 2)
        d.polygon([tuple(q) for q in hull], fill=RGB[c], outline="black")
        if c in lay.get("targets", [lay["target"]]):
            d.line([tuple(q) for q in hull] + [tuple(hull[0])], fill=(255, 255, 255), width=3)
    placed = []
    for c in sorted(lay["blocks"], key=lambda c: num[c]):     # labels last, moved along the block when they collide
        p, R = ut.frame(c)
        q = p[:2].copy()
        for s in (0.0, 0.02, -0.02, 0.027, -0.027):
            q = p[:2] + s * R[:2, 0]
            if all(np.linalg.norm(q - o) > 0.012 for o in placed):
                break
        placed.append(q)
        cx, cy = to_px(*q)
        d.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill="white", outline="black")
        d.text((cx - 5, cy - 9), str(num[c]), fill="black", font=fs)
    return img


def outline(img, ut, blocks, width=2):
    """Draw the edges of each block, as the base camera sees them, on a BGR frame."""
    g = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    for c in blocks:
        p, R = ut.frame(c)
        uv = ut.tw.project(p + (g * ut.half) @ R.T, ut.tw.base_camera)
        for i in range(8):
            for j in range(i + 1, 8):
                if np.abs(g[i] - g[j]).sum() == 2:
                    cv2.line(img, tuple(int(v) for v in uv[i]), tuple(int(v) for v in uv[j]), RGB[c][::-1], width,
                             cv2.LINE_AA)
    return img


def grid(img, ut, step=0.05):
    """The Letter-size workspace rectangle on the table, with a line every 5 cm from its far and left edges."""
    sx, sy = ut.tw.cfg["sheet"]["size"]
    seg = [((sx / 2 - d, sy / 2), (sx / 2 - d, -sy / 2)) for d in np.arange(0, sx + 1e-6, step)]
    seg += [((sx / 2, sy / 2 - d), (-sx / 2, sy / 2 - d)) for d in np.arange(0, sy + 1e-6, step)]
    seg += [((-sx / 2, sy / 2), (-sx / 2, -sy / 2)), ((sx / 2, -sy / 2), (-sx / 2, -sy / 2))]
    for k, (a, b) in enumerate(seg):
        pts = np.linspace([*a, 0.0], [*b, 0.0], 20)
        uv = ut.tw.project(pts, ut.tw.base_camera)
        if not np.isnan(uv).any():
            cv2.polylines(img, [np.int32(uv)], False, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_sheet(ut, lay, role):
    tw = ut.tw
    covers = [d[0] for d in lay["drops"]]
    state = (np.array(lay["qpos"]), lay["present"])
    ut.restore(state)
    rows = describe(ut, lay)
    top = top_view(ut, lay, rows)
    final = Image.fromarray(tw.render(cam=tw.base_camera))
    show_only(ut, state, [c for c in lay["blocks"] if c not in covers])
    before = Image.fromarray(tw.render(cam=tw.base_camera)) if covers else None
    ut.restore(state)
    cw = 560
    ch = int(cw * final.height / final.width)
    W = top.width + cw + 50
    H = max(top.height + 70, 110 + 2 * (ch + 40)) + 30 * len(rows) + 90
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    targets = lay.get("targets", [lay["target"]])
    d.text((20, 8), f"{lay['id']}  [{role}]   target{'s, in this order' * (len(targets) > 1)}: "
                    f"{' then '.join(t.upper() for t in targets)}", fill="black", font=font(24))
    d.text((20, 40), f"{ {**KINDS, **KINDS2}[lay['kind']][2]}; occlusion: {lay['occlusion']}; best order: "
                     f"{' → '.join(lay['plan'])} ({lay['optimal_steps']} step{'s' * (lay['optimal_steps'] > 1)})",
           fill="black", font=font(18))
    sheet.paste(top, (0, 70))
    x0, y0 = top.width + 20, 80
    if before is not None:
        d.text((x0, y0), "arm-camera view, table blocks only:", fill="black", font=font(17))
        sheet.paste(before.resize((cw, ch)), (x0, y0 + 26))
        y0 += ch + 40
    d.text((x0, y0), "arm-camera view, finished scene:", fill="black", font=font(17))
    sheet.paste(final.resize((cw, ch)), (x0, y0 + 26))
    y = max(top.height + 70, y0 + ch + 40)
    d.text((20, y), "Build in this order (numbers = the circles in the top view). Positions are block centres.",
           fill=(60, 60, 60), font=font(15))
    for k, r in enumerate(rows):
        d.text((20, y + 28 + 30 * k), r["text"], fill="black", font=font(17))
    return sheet, rows


# ── main ──

def check(args):
    """A real frame next to the layout's expected view, with the expected block edges drawn on both."""
    frame = Path(args.frame)
    if frame.is_dir():                                 # a server session folder: its newest frame
        frame = max(frame.rglob("frame.jpg"), key=lambda f: f.stat().st_mtime)
    real = cv2.imread(str(frame))
    if real is None:
        raise SystemExit(f"cannot read {frame}")
    ut = UnveilTwin(unveil_args([]))
    blocks = []
    if args.check != "grid":
        lays = {l["id"]: l for l in json.load(open(Path(args.out) / "layouts.json"))["layouts"]}
        lay = lays[args.check]
        blocks = lay["blocks"]
        ut.restore((np.array(lay["qpos"]), lay["present"]))
    else:
        for c in ut.present():
            ut.remove(c, settle=False)
    twin = cv2.cvtColor(ut.tw.render(cam=ut.tw.base_camera), cv2.COLOR_RGB2BGR)
    real = cv2.resize(real, (twin.shape[1], twin.shape[0]))
    out = np.hstack([outline(grid(real.copy(), ut), ut, blocks), outline(grid(twin.copy(), ut), ut, blocks, 1)])
    dst = args.check_out or str(frame.with_name(f"check_{args.check}.jpg"))
    cv2.imwrite(dst, out)
    print(dst)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="twin/real_eval_layouts")
    ap.add_argument("--n", type=int, nargs=3, default=[10, 6, 4], metavar=("C1", "C2", "F"), help="eval layouts")
    ap.add_argument("--spares", type=int, nargs=3, default=[2, 2, 1], metavar=("C1", "C2", "F"))
    ap.add_argument("--n2", type=int, nargs=2, default=[6, 4], metavar=("T2a", "T2b"), help="Task 2 eval layouts")
    ap.add_argument("--spares2", type=int, nargs=2, default=[1, 1], metavar=("T2a", "T2b"))
    ap.add_argument("--seed", type=int, default=20261006, help="training used seeds below 10000, twin/eval.py 12345")
    ap.add_argument("--robust", type=int, default=8, help="noisy rebuilds that must give the same answer")
    ap.add_argument("--noise_pos", type=float, default=0.004, help="m, std of the hand-placement error")
    ap.add_argument("--noise_yaw", type=float, default=5.0, help="deg")
    ap.add_argument("--margin", type=float, default=12.0, help="px from the camera image border")
    ap.add_argument("--view_margin", type=float, default=30.0, help="px from the border of the 400x400 workspace view")
    ap.add_argument("--max_tilt", type=float, default=40.0, help="deg; steeper blocks are hard to place by hand")
    ap.add_argument("--check", default=None, metavar="ID",
                    help="draw this layout's expected block edges (or only the workspace 'grid') on --frame and exit")
    ap.add_argument("--frame", default=None)
    ap.add_argument("--check_out", default=None)
    args = ap.parse_args(argv)
    if args.check:
        return check(args)

    ut = UnveilTwin(unveil_args([]))
    colors = ut.tw.colors
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layouts, tried = [], {}
    for kind, n_eval, n_spare in zip(KINDS, args.n, args.spares):
        done = 0
        while done < n_eval + n_spare:
            # The colours take turns as the target and as the covering block, so no colour is the answer more often
            # than another. C1 alternates 3 and 4 blocks, C2 and F use 4 (as the real test set).
            n_blocks = 4 if kind != "C1" or done % 2 else 3
            target = colors[(done + list(KINDS).index(kind)) % len(colors)]
            rest = [c for c in colors if c != target]
            turn = (done + done // len(colors)) % len(rest)
            lay = sample(ut, rng, kind, target, (rest[turn:] + rest[:turn])[:n_blocks - 1])
            tried[kind] = tried.get(kind, 0) + 1
            if lay is None or not accept(ut, lay, args):
                continue
            done += 1
            lay["id"] = f"{kind}-{done:02d}"
            lay["role"] = "EVAL" if done <= n_eval else "SPARE"
            sheet, rows = make_sheet(ut, lay, lay["role"])
            sheet.save(out / f"{lay['id']}.png")
            lay["placement"] = rows
            layouts.append(lay)
            print(f"{lay['id']} [{lay['role']}] target {lay['target']}, {len(lay['blocks'])} blocks, "
                  f"{lay['occlusion']}, plan {lay['plan']}", flush=True)
    for kind, n_eval, n_spare in zip(KINDS2, args.n2, args.spares2):
        done = 0
        while done < n_eval + n_spare:
            a_t = colors[(done + list(KINDS2).index(kind)) % len(colors)]
            rest = [c for c in colors if c != a_t]
            turn = (done + done // len(colors)) % len(rest)
            rest = rest[turn:] + rest[:turn]
            lay = sample2(ut, rng, kind, a_t, rest[0], rest[1:])
            tried[kind] = tried.get(kind, 0) + 1
            if lay is None or not accept2(ut, lay, args):
                continue
            done += 1
            lay["id"] = f"{kind}-{done:02d}"
            lay["role"] = "EVAL" if done <= n_eval else "SPARE"
            sheet, rows = make_sheet(ut, lay, lay["role"])
            sheet.save(out / f"{lay['id']}.png")
            lay["placement"] = rows
            layouts.append(lay)
            print(f"{lay['id']} [{lay['role']}] targets {lay['targets']}, {lay['occlusion']}, plan {lay['plan']}",
                  flush=True)
    ut.tw.close()
    for name, kinds in (("layouts.pdf", KINDS), ("layouts_task2.pdf", KINDS2)):     # one sheet per page, to print
        pages = [Image.open(out / f"{lay['id']}.png").convert("RGB") for lay in layouts if lay["kind"] in kinds]
        if pages:
            pages[0].save(out / name, save_all=True, append_images=pages[1:])
    with open(out / "layouts.json", "w") as f:
        json.dump({"config": vars(args), "candidates_tried": tried, "layouts": layouts}, f, indent=1)
    count = {k: sum(l["kind"] == k and l["role"] == "EVAL" for l in layouts) for k in (*KINDS, *KINDS2)}
    spread = lambda l: ((int(l["id"].split("-")[1]) - 0.5) / max(count[l["kind"]], 1), l["kind"])
    notes = {"EVAL": "In run order: the kinds are spread evenly over the session.",
             "SPARE": "When a layout cannot be built or reached, use the next unused spare of the same kind, for "
                      "every method."}
    for kinds, stem, task in ((KINDS, "scenes", "Task 1"), (KINDS2, "scenes_task2", "Task 2 (two targets in order)")):
        for role, suffix in (("EVAL", ""), ("SPARE", "_spares")):
            with open(out / f"{stem}{suffix}.yaml", "w") as f:
                f.write(f"# {task} scenes for unveiler/unveiler_session.py, one per layout sheet "
                        "(twin/make_eval_layouts.py).\n# The twin search's answer for each scene is in layouts.json "
                        f"(optimal_steps, first_choices, plan).\n# {notes[role]}\n")
                for lay in sorted(layouts, key=spread):
                    if lay["role"] != role or lay["kind"] not in kinds:
                        continue
                    blocks = f"blocks: [{', '.join(lay['blocks'])}]"
                    if "targets" in lay:
                        f.write(f"- {{id: {lay['id']}, task: ordered, targets: [{', '.join(lay['targets'])}], {blocks}}}\n")
                    else:
                        f.write(f"- {{id: {lay['id']}, density: \"{len(lay['blocks'])}\", occlusion: "
                                f"{lay['occlusion']}, target: {lay['target']}, {blocks}}}\n")
    print(f"{out}/layouts.json, {out}/scenes.yaml; candidates tried {tried}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
