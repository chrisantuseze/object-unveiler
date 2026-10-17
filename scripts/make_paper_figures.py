"""Real-robot figures for save/paper/root.tex, built from the frames saved by the robot client.

    python scripts/make_paper_figures.py        # writes save/paper/real_*.pdf

Frames come from save/real_eval/results_jetson/<session>/ep_NNN/ (the arm camera, 640x480) and the twin renders from the
layout sheets in twin/real_eval_layouts/. The frames are dark, so they are brightened with one fixed gamma for print;
nothing else in them is changed. Outlines are the masks the server used: magenta for the block selected for removal,
green for a target.
"""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "save/real_eval/results_jetson"
SHEETS = ROOT / "twin/real_eval_layouts"
OUT = ROOT / "save/paper"
GAMMA = 0.6
OBSTACLE, TARGET = (255, 0, 160), (40, 220, 60)       # RGB
# arm-camera view of the finished scene on a 1210x1240 sheet (y0, y1, x0, x1); free-target sheets have one view only
SHEET_VIEW, SHEET_VIEW_FREE = (565, 985, 620, 1182), (106, 526, 620, 1182)

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7, "axes.titlepad": 3})


def frame(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (255 * img ** GAMMA).astype(np.uint8)


def outline(img, mask_path, colour, width=4):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return img
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(img.copy(), contours, -1, colour, width)


def episode(session, scene):
    for line in open(RES / session / "episodes.jsonl"):
        e = json.loads(line)
        if e["scene"] == scene:
            return e, RES / session / f"ep_{e['episode']:03d}"
    raise KeyError((session, scene))


def show(ax, img, title, colour="black"):
    ax.imshow(img)
    ax.set_title(title, color=colour)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def rollout(session="task2_ours", scene="T2a-01"):
    """One Task 2 episode: every selection, in order."""
    e, d = episode(session, scene)
    steps = e["steps"]
    fig, axes = plt.subplots(1, len(steps), figsize=(7.1, 1.75))
    for ax, s in zip(axes, steps):
        sd = d / f"step_{s['step']:02d}"
        p = json.load(open(sd / "reply.json"))["probs"][s["chosen_index"]]
        img = outline(frame(sd / "frame.jpg"), sd / "chosen_mask.png", TARGET if s["is_target"] else OBSTACLE)
        verb = "grasp target" if s["is_target"] else "remove"
        show(ax, img, f"Step {s['step'] + 1}: {verb} {s['colour']}\n$p$ = {p:.2f}")
    fig.subplots_adjust(left=0.003, right=0.997, top=0.85, bottom=0.01, wspace=0.03)
    fig.savefig(OUT / "real_rollout.pdf", dpi=200)


def scenes(items=(("C1-03", "task1_ours", "Task 1: one cover"), ("C2-02", "task1_ours", "Task 1: two covers"),
                  ("F-02", "task1_ours", "Task 1: free target"), ("T2a-05", "task2_ours", "Task 2: two targets"))):
    """Each evaluation condition: the layout rendered in the twin, above the same layout built on the table."""
    fig, axes = plt.subplots(2, len(items), figsize=(7.1, 2.75))
    for k, (scene, session, title) in enumerate(items):
        y0, y1, x0, x1 = SHEET_VIEW_FREE if scene.startswith("F") else SHEET_VIEW
        sheet = cv2.cvtColor(cv2.imread(str(SHEETS / f"{scene}.png")), cv2.COLOR_BGR2RGB)
        show(axes[0, k], sheet[y0:y1, x0:x1], title)
        show(axes[1, k], frame(RES / session / "layouts" / f"{scene}.jpg"), "")
    axes[0, 0].set_ylabel("Twin"), axes[1, 0].set_ylabel("Real")
    fig.subplots_adjust(left=0.03, right=0.997, top=0.93, bottom=0.01, wspace=0.03, hspace=0.04)
    fig.savefig(OUT / "real_scenes.pdf", dpi=200)


def first_choice(scene="T2b-01", methods=(("ours", "Unveiler (ours)"), ("il", "IL"), ("heuristic", "Heur"),
                                          ("gpt4o", "GPT-4o")), correct=("blue",)):
    """The first selection of every method on one Task 2 scene."""
    fig, axes = plt.subplots(1, len(methods), figsize=(7.1, 1.55))
    for ax, (m, name) in zip(axes, methods):
        e, d = episode(f"task2_{m}", scene)
        s, sd = e["steps"][0], d / "step_00"
        img = outline(frame(sd / "frame.jpg"), sd / "chosen_mask.png", TARGET if s["is_target"] else OBSTACLE)
        ok = s["colour"] in correct
        what = f"grasps {s['colour']} (covered target)" if s["is_target"] else f"removes {s['colour']}"
        show(ax, img, f"{name}: {what}", "#1a7f37" if ok else "#b42318")
    fig.subplots_adjust(left=0.003, right=0.997, top=0.90, bottom=0.01, wspace=0.03)
    fig.savefig(OUT / "real_first_choice.pdf", dpi=200)


def probabilities(items=(("C2-02", 0, "Two covers"), ("C1-04", 0, "One cover"), ("F-02", 0, "Free target")),
                  session="task1_ours"):
    """The SRE's probability for every segmented object, on real frames (the target's reference mask in green)."""
    fig, axes = plt.subplots(1, len(items), figsize=(7.1, 1.95))
    for ax, (scene, step, title) in zip(axes, items):
        e, d = episode(session, scene)
        sd = d / f"step_{step:02d}"
        reply = json.load(open(sd / "reply.json"))
        img = outline(frame(sd / "frame.jpg"), d / "ref_mask.png", TARGET, 3)
        show(ax, img, f"{title} (target: {e['target_colour']})")
        for o, pr in zip(reply["objects"], reply["probs"]):
            x0, y0, x1, y1 = o["bbox"]
            best = o["index"] == reply["chosen_index"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.4 if best else 0.7,
                                       ec="#ff00a0" if best else "white", ls="-" if best else "--"))
            ax.text(x0 + 4, y0 + 6, f"{pr:.2f}", va="top", fontsize=7, color="white",
                    bbox=dict(fc="#ff00a0" if best else "#444444", ec="none", pad=1.2))
    fig.subplots_adjust(left=0.003, right=0.997, top=0.92, bottom=0.01, wspace=0.03)
    fig.savefig(OUT / "real_probabilities.pdf", dpi=200)


def clip_frame(name, seconds, width=640):
    """One frame of an external-camera clip (save/real_eval/results_jetson/IMG_*.MOV), cropped to the robot and table."""
    cap = cv2.VideoCapture(str(RES / name))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, f = cap.read()
    cap.release()
    assert ok, (name, seconds)
    f = f[40:1040, 80:1860]
    return cv2.cvtColor(cv2.resize(f, (width, int(width * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)


def external_rollout(clip="IMG_8713.MOV", frames=((0, "Start"),
                                                  (12.4, "1: removes blue"), (40.3, "2: grasps red (target 1)"),
                                                  (83.7, "3: removes yellow"), (111.6, "4: grasps green (target 2)"),
                                                  (130.2, "End: both targets in the bin")),
                     out="real_external_rollout.pdf"):
    """The setup from outside: Unveiler on layout T2a-01 (a filmed run, not an evaluation episode)."""
    fig, axes = plt.subplots(2, 3, figsize=(5.0, 2.15))
    for ax, (t, title) in zip(axes.ravel(), frames):
        show(ax, clip_frame(clip, t), title)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.93, bottom=0.01, wspace=0.03, hspace=0.16)
    fig.savefig(OUT / out, dpi=220)


def external_task1():
    """Fig. 1: Unveiler on layout C2-02 from outside (a filmed run); the red block covers nothing and is left."""
    external_rollout("IMG_8711.MOV", ((0.5, "Start (target: blue)"),
                                      (15.2, "1: removes yellow"), (43.1, "2: removes green"),
                                      (58.5, "Blue target exposed"), (73.6, "3: grasps blue (target)"),
                                      (85.5, "End: red left on the table")), "real_external_task1.pdf")


def external_comparison(rows=(("Unveiler (ours)", "IMG_8715.MOV", ((15.5, "removes blue"), (49.6, "grasps green"),
                                                                    (80.6, "grasps red"))),
                              ("GPT-4o", "IMG_8719.MOV", ((1.2, "start"), (13.2, "grasps at green, under blue"),
                                                          (24.0, "green not retrieved"))))):
    """Layout T2b-01 from outside, for Unveiler and GPT-4o (filmed runs, not evaluation episodes)."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(5.0, 2.15))
    for r, (name, clip, frames) in enumerate(rows):
        for ax, (t, title) in zip(axes[r], frames):
            show(ax, clip_frame(clip, t), f"{name}: {title}" if ax is axes[r][0] else title)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.93, bottom=0.01, wspace=0.03, hspace=0.16)
    fig.savefig(OUT / "real_external_comparison.pdf", dpi=220)


if __name__ == "__main__":
    external_task1()
    scenes()
    first_choice()
    probabilities()
    external_rollout()
    external_comparison()
