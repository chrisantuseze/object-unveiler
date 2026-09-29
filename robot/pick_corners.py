#!/usr/bin/env python3
"""Click the four workspace corners on a camera frame and print the matching ``--warp`` arguments.

    python -m robot.pick_corners save/real_eval/<session>/episode_0001/step_00_sre/frame.jpg

Click in order: top-left, top-right, bottom-right, bottom-left (as the workspace should appear in the straight-down
view). Backspace undoes the last click, Enter accepts, Esc quits. After four clicks a preview of the rectified
400x400 view opens beside the frame; check that the blocks look square-ish and the workspace edges are straight.

Headless machines (no $DISPLAY, e.g. over SSH): the tool auto-detects the largest bright quadrilateral (the white
workspace sheet), writes ``<preview-out>`` with the detected corners, a pixel grid for reading coordinates, and the
rectified view, and prints the suggested ``--warp``. Open the preview in any image viewer (VS Code works). If the
detection is off, read better corners off the grid and re-run with ``--corners TLx TLy TRx TRy BRx BRy BLx BLy`` to
check them.
"""

import argparse
import os
import sys

import cv2
import numpy as np

SIM_SIZE = 400
ORDER = ("top-left", "top-right", "bottom-right", "bottom-left")


def rectify(frame, corners):
    src = np.float32(corners).reshape(4, 2)
    dst = np.float32([[0, 0], [SIM_SIZE, 0], [SIM_SIZE, SIM_SIZE], [0, SIM_SIZE]])
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (SIM_SIZE, SIM_SIZE))


def draw(frame, pts):
    out = frame.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(out, ORDER[i], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if len(pts) > 1:
        cv2.polylines(out, [np.int32(pts)], len(pts) == 4, (0, 255, 0), 2)
    if len(pts) < 4:
        cv2.putText(out, f"click {ORDER[len(pts)]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def order_corners(pts):
    """Any 4 points -> TL, TR, BR, BL."""
    pts = np.float32(pts).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]]


def detect_sheet(frame):
    """Corners of the largest bright quadrilateral (the workspace sheet), or None."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.1 * frame.shape[0] * frame.shape[1]:
        return None
    hull = cv2.convexHull(c)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return [list(map(float, p)) for p in order_corners(approx)]
    return None


def draw_grid(img, step=40):
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (200, 200, 200), 1)
        cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (200, 200, 200), 1)
        cv2.putText(out, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--corners", type=float, nargs=8, default=None, help="skip clicking: TL TR BR BL as x y")
    ap.add_argument("--preview-out", default="warp_preview.jpg")
    args = ap.parse_args(argv)

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not read {args.image}")

    headless = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    if args.corners:
        pts = np.float32(args.corners).reshape(4, 2).tolist()
    elif headless:
        pts = detect_sheet(frame)
        if pts is None:
            cv2.imwrite(args.preview_out, draw_grid(frame))
            print(f"no display and no sheet detected; read the corners off the grid in {args.preview_out} "
                  f"and re-run with --corners TLx TLy TRx TRy BRx BRy BLx BLy")
            return 1
        print("no display: auto-detected the workspace sheet (check the preview)")
    else:
        pts = []
        win = "pick corners (Enter = accept, Backspace = undo, Esc = quit)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, lambda ev, x, y, *_: pts.append([x, y]) if ev == cv2.EVENT_LBUTTONDOWN
                             and len(pts) < 4 else None)
        while True:
            view = draw(frame, pts)
            if len(pts) == 4:
                view = np.hstack([view, cv2.copyMakeBorder(rectify(frame, pts), 0, max(0, frame.shape[0] - SIM_SIZE),
                                                           0, 0, cv2.BORDER_CONSTANT)[:frame.shape[0]]])
            cv2.imshow(win, view)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return 1
            if key == 8 and pts:
                pts.pop()
            if key in (13, 10) and len(pts) == 4:
                break
        cv2.destroyAllWindows()

    cv2.imwrite(args.preview_out, np.hstack([draw(draw_grid(frame), pts),
                                             cv2.resize(rectify(frame, pts), (frame.shape[0], frame.shape[0]))]))
    flat = " ".join(str(int(round(v))) for p in pts for v in p)
    print(f"preview: {args.preview_out}")
    print(f"--warp {flat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
