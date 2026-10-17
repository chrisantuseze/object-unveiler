"""Supplementary video: filmed re-runs of four evaluation layouts, with each selection overlaid from the robot's logs.

    python scripts/make_video.py                 # writes save/paper/jint/video/ESM_1.mp4 (Online Resource 1)
    python scripts/make_video.py --preview DIR   # a few composed stills per segment instead of the video

Footage is save/real_eval/results_jetson/IMG_*.MOV (one clip per episode, external camera). The panels come from the same
runs: the arm-camera frame, the selected block's mask and the SRE's probability are read from
save/real_eval/results_jetson/film_*/ep_NNN/step_NN/, and the time of each selection from the server's logs in
save/real_eval/film_*/. A panel switches when its selection was made, counted from the start of the clip.
Needs an ffmpeg binary (pip install imageio-ffmpeg).
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "save/real_eval/results_jetson"
LOGS = ROOT / "save/real_eval"
OUT = ROOT / "save/paper/jint/video"
W, H, FPS = 1280, 720, 30
BG, FG, DIM = (18, 18, 20), (240, 240, 240), (150, 150, 155)
GOOD, BAD, PICK, TARGET = (70, 200, 110), (235, 90, 80), (255, 0, 160), (40, 220, 60)      # RGB
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
GAMMA = 0.6
TAIL = 3.0            # seconds usually filmed after an episode ends

TITLE = "Learning Object-Centric Spatial Reasoning for\nSequential Manipulation in Cluttered Environments"
AUTHORS = "Chrisantus Eze, Ryan C. Julian, Christopher Crick"
AFFIL = "Oklahoma State University  ·  Google DeepMind"
JOURNAL = "Journal of Intelligent & Robotic Systems"
# Springer asks for the corresponding author's affiliation and e-mail in every supplementary file.
CORRESPONDING = "Corresponding author: Chrisantus Eze, Oklahoma State University"
CORRESPONDING_EMAIL = "chrisantus.eze@okstate.edu"

# clip, session, scene, method label, outcome text, outcome is a success
CLIPS = {
    "ours_T2a": ("IMG_8713.MOV", "film_task2_ours", "T2a-01", "Unveiler (ours)", "Both targets retrieved, in order", True),
    "ours_T2b": ("IMG_8715.MOV", "film_task2_ours", "T2b-01", "Unveiler (ours)", "Both targets retrieved, in order", True),
    "gpt_T2b": ("IMG_8719.MOV", "film_task2_gpt4o", "T2b-01", "GPT-4o",
                "Fails: grasps the green target while\nthe blue block still covers it", False),
    "ours_C2": ("IMG_8711.MOV", "film_task1_ours", "C2-02", "Unveiler (ours)", "Target retrieved", True),
    "gpt_C2": ("IMG_8716.MOV", "film_task1_gpt4o", "C2-02", "GPT-4o",
               "Fails: removes red, which covers nothing,\nthen selects the still-covered target", False),
    "ours_F": ("IMG_8712.MOV", "film_task1_ours", "F-02", "Unveiler (ours)", "Target retrieved", True),
    "gpt_F": ("IMG_8717.MOV", "film_task1_gpt4o", "F-02", "GPT-4o", "Target retrieved", True),
}


def font(size, bold=False):
    return ImageFont.truetype(FONT % ("-Bold" if bold else ""), size)


def text(img, xy, s, size=22, fill=FG, bold=False, anchor="la", spacing=6):
    """Draw text on a BGR array (in place) with PIL."""
    pil = Image.fromarray(img[:, :, ::-1])
    ImageDraw.Draw(pil).multiline_text(xy, s, font=font(size, bold), fill=fill, anchor=anchor, spacing=spacing,
                                       align="center" if anchor[0] == "m" else "left")
    img[:] = np.asarray(pil)[:, :, ::-1]


def canvas():
    return np.full((H, W, 3), BG[::-1], np.uint8)


class Episode:
    """One filmed episode: its footage and what the logs say was selected, and when."""

    def __init__(self, key):
        mov, session, scene, self.label, self.outcome, self.ok = CLIPS[key]
        self.path = str(RES / mov)
        for line in open(RES / session / "episodes.jsonl"):
            e = json.loads(line)
            if e["scene"] == scene:
                break
        self.e, self.dir = e, RES / session / f"ep_{e['episode']:03d}"
        self.targets = e.get("targets") or [e["target_colour"]]
        method = "gpt4o" if "gpt4o" in session else "ours"
        times = [os.path.getmtime(LOGS / session / f"episode_{e['episode']:04d}" / f"step_{s['step']:02d}_{method}" /
                                  "reply.json") for s in e["steps"]]
        self.offsets = [t - times[0] for t in times]            # seconds after the first selection
        self.panels = [self._panel(s) for s in e["steps"]]
        self.lines = [self._line(s) for s in e["steps"]]
        cap = cv2.VideoCapture(self.path)
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # Source frame of the first selection. Filming started as the arm began to move, so it is the first frame,
        # unless the clip is shorter than the episode: then filming started late by the difference (plus the usual
        # few seconds filmed after the end).
        self.onset = int(FPS * min(0.0, self.n / FPS - e["duration_s"] - TAIL))

    def _panel(self, s):
        sd = self.dir / f"step_{s['step']:02d}"
        img = cv2.imread(str(sd / "frame.jpg")).astype(np.float32) / 255.0
        img = (255 * img ** GAMMA).astype(np.uint8)
        mask = cv2.imread(str(sd / "chosen_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 0:
            cnt, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnt, -1, (TARGET if s["is_target"] else PICK)[::-1], 5)
        return img

    def _line(self, s):
        if s.get("chosen_index") is None or s.get("colour") is None:
            return "selects the target (still covered)", None
        r = json.load(open(self.dir / f"step_{s['step']:02d}" / "reply.json"))
        p = r["probs"][s["chosen_index"]] if r.get("probs") else None
        verb = "grasp target:" if s["is_target"] else "remove"
        return f"{verb} {s['colour']}", p

    def step_at(self, src_frame):
        """Index of the latest selection made by this source frame."""
        t = (src_frame - self.onset) / FPS
        return max(0, sum(o <= t + 1.0 for o in self.offsets) - 1)


def draw_steps(img, ep, k, x, y, size=20, done=False):
    """The selections so far, the current one highlighted."""
    for i in range(k + 1):
        line, p = ep.lines[i]
        cur = i == k and not done
        s = f"{i + 1}.  {line}" + (f"     p = {p:.2f}" if p is not None else "")
        text(img, (x, y + i * (size + 12)), s, size, FG if cur else DIM, bold=cur)


def banner(img, ep, x0, y0, x1, y1):
    col = GOOD if ep.ok else BAD
    cv2.rectangle(img, (x0, y0), (x1, y1), col[::-1], -1)
    size = 21
    while max(font(size, True).getlength(t) for t in ep.outcome.split("\n")) > x1 - x0 - 16:
        size -= 1
    text(img, ((x0 + x1) // 2, (y0 + y1) // 2), ep.outcome, size, (10, 10, 10), bold=True, anchor="mm")


def header(img, title, sub):
    text(img, (28, 22), title, 30, bold=True)
    text(img, (28, 64), sub, 20, DIM)


def play(eps, compose, speed, emit, hold=2.0, stills=None):
    """Run the episodes' clips side by side at `speed`; a clip that ends holds its last frame."""
    caps = [cv2.VideoCapture(e.path) for e in eps]
    last, src = [None] * len(eps), 0
    total = max(e.n for e in eps)
    while src < total:
        for j, (c, e) in enumerate(zip(caps, eps)):
            for _ in range(speed):
                if not c.grab():
                    break
            else:
                ok, f = c.retrieve()
                if ok:
                    last[j] = f
        src += speed
        frame = compose(last, [min(src, e.n) for e in eps], [src >= e.n for e in eps])
        if stills is not None:
            if src // speed in stills:
                emit(frame)
        else:
            emit(frame)
    final = compose(last, [e.n for e in eps], [True] * len(eps))
    for _ in range(1 if stills is not None else int(hold * FPS)):
        emit(final)
    for c in caps:
        c.release()


def single(ep, title, sub, note, speed, emit, stills=None):
    def compose(frames, pos, ended):
        img = canvas()
        header(img, title, sub)
        img[110:110 + 477, 28:28 + 848] = cv2.resize(frames[0], (848, 477), interpolation=cv2.INTER_AREA)
        k = ep.step_at(pos[0])
        text(img, (904, 106), f"{ep.label}  ·  arm camera", 18, DIM)
        img[134:134 + 261, 904:904 + 348] = cv2.resize(ep.panels[k], (348, 261), interpolation=cv2.INTER_AREA)
        draw_steps(img, ep, k, 904, 412, 19, done=ended[0])
        if ended[0]:
            banner(img, ep, 904, 540, 1252, 587)
            text(img, (904, 600), note, 16, DIM)
        text(img, (28, 600), f"{speed}× speed", 18, DIM)
        return img
    play([ep], compose, speed, emit, stills=stills)


def dual(a, b, title, sub, speed, emit, stills=None):
    def compose(frames, pos, ended):
        img = canvas()
        header(img, title, sub)
        for j, (ep, x) in enumerate(((a, 20), (b, 648))):
            text(img, (x, 104), ep.label, 22, bold=True)
            img[136:136 + 344, x:x + 612] = cv2.resize(frames[j], (612, 344), interpolation=cv2.INTER_AREA)
            k = ep.step_at(pos[j])
            img[494:494 + 150, x:x + 200] = cv2.resize(ep.panels[k], (200, 150), interpolation=cv2.INTER_AREA)
            draw_steps(img, ep, k, x + 216, 494, 17, done=ended[j])
            if ended[j]:
                banner(img, ep, x, 652, x + 612, 708)
        text(img, (W - 28, 30), f"{speed}× speed", 18, DIM, anchor="ra")
        return img
    play([a, b], compose, speed, emit, stills=stills)


def card(lines, seconds, emit, stills=None):
    """A text card: (text, size, colour, bold, gap after) per line, centred."""
    img = canvas()
    heights = [len(t.split("\n")) * (s + 8) + g for t, s, _, _, g in lines]
    y = (H - sum(heights)) // 2
    for (t, s, col, bold, _), h in zip(lines, heights):
        text(img, (W // 2, y), t, s, col, bold=bold, anchor="ma", spacing=8)
        y += h
    for _ in range(1 if stills is not None else int(seconds * FPS)):
        emit(img)


def results_card(seconds, emit, stills=None):
    img = canvas()
    text(img, (W // 2, 70), "Real-robot results (10 scenes per task)", 32, bold=True, anchor="ma")
    cols = [330, 560, 730, 930, 1100]
    for x, t in zip(cols[1:], ("success", "correct\nselections", "success", "correct\nselections")):
        text(img, (x, 205), t, 20, DIM, anchor="ma")
    text(img, ((cols[1] + cols[2]) // 2, 160), "Task 1: one target", 22, bold=True, anchor="ma")
    text(img, ((cols[3] + cols[4]) // 2, 160), "Task 2: two targets, in order", 22, bold=True, anchor="ma")
    rows = [("GPT-4o", "30%", "45.0%", "20%", "55.6%"), ("Heuristic", "40%", "70.6%", "20%", "71.4%"),
            ("Imitation only", "40%", "69.2%", "30%", "75.0%"), ("Unveiler (ours)", "70%", "85.7%", "90%", "100%")]
    for i, r in enumerate(rows):
        y, ours = 285 + i * 62, i == len(rows) - 1
        text(img, (cols[0] - 150, y), r[0], 24, GOOD if ours else FG, bold=ours)
        for x, v in zip(cols[1:], r[1:]):
            text(img, (x, y), v, 24, GOOD if ours else FG, bold=ours, anchor="ma")
    for _ in range(1 if stills is not None else int(seconds * FPS)):
        emit(img)


def build(emit, stills=None):
    E = {k: Episode(k) for k in CLIPS}
    for k, e in E.items():
        print(f"{k}: {e.n / FPS:.0f} s, first selection at {e.onset / FPS:.1f} s, selections at "
              f"{[round(o) for o in e.offsets]} s after the first")
    contact = CORRESPONDING + (f", {CORRESPONDING_EMAIL}" if CORRESPONDING_EMAIL else "")
    card([(TITLE, 36, FG, True, 40), (AUTHORS, 24, FG, False, 10), (AFFIL, 20, DIM, False, 44),
          (JOURNAL, 22, FG, False, 10), (contact, 18, DIM, False, 0)], 6, emit, stills)
    card([("Retrieving a covered target", 34, FG, True, 34),
          ("A Spatial Relationship Encoder selects which object to remove next.\n"
           "The robot removes it with its own pick-and-place, and the encoder selects again.", 23, FG, False, 26),
          ("The encoder is trained only in simulation, in a digital twin of this workspace.\n"
           "No real-robot data is used.", 23, FG, False, 26),
          ("Magenta outline: block selected for removal.  Green outline: target being grasped.\n"
           "p: the encoder's probability for its selection.", 20, DIM, False, 0)], 9, emit, stills)
    s = stills
    single(E["ours_T2a"], "Task 2: retrieve red, then green", "Each target lies under its own block.",
           "GPT-4o also solved this layout when filmed.", 5, emit, s)
    dual(E["ours_T2b"], E["gpt_T2b"], "Task 2: retrieve green, then red", "One blue block lies across both targets.",
         5, emit, s)
    dual(E["ours_C2"], E["gpt_C2"], "Task 1: retrieve blue", "The target lies under two blocks (yellow and green).",
         5, emit, s)
    dual(E["ours_F"], E["gpt_F"], "Task 1: retrieve yellow", "The target is free: nothing needs to be removed.",
         4, emit, s)
    results_card(9, emit, stills)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="write a few stills per segment to this folder instead of the video")
    args = ap.parse_args()
    if args.preview:
        out, n = Path(args.preview), [0]
        out.mkdir(parents=True, exist_ok=True)

        def emit(f):
            cv2.imwrite(str(out / f"still_{n[0]:03d}.jpg"), f)
            n[0] += 1
        build(emit, stills={60, 260, 520})
        return
    import imageio_ffmpeg
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "ESM_1.mp4"
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                           "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                           "yuv420p", "-crf", "23", "-preset", "medium", "-movflags", "+faststart", str(dst)],
                          stdin=subprocess.PIPE)
    n = [0]

    def emit(f):
        ff.stdin.write(f.tobytes())
        n[0] += 1
    build(emit)
    ff.stdin.close()
    ff.wait()
    print(f"{dst}: {n[0] / FPS:.0f} s, {dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
