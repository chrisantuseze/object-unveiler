#!/usr/bin/env python3
"""Mock Jetson: sends the requests the robot's episode runner will send, over rosbridge, and checks the replies.

Run against the fake bridge on one machine (three terminals, or see robot/README.md for the one-liner):
    python -m robot.fake_rosbridge --port 9090
    python -m robot.server --jetson-ip 127.0.0.1 --crop 120 40 520 440
    python -m robot.mock_client --jetson-ip 127.0.0.1

or against the real Jetson's rosbridge from the lab PC (server running), to test the network path without the arm:
    python -m robot.mock_client --jetson-ip 192.168.0.8

The frame is a sim top-down scene placed in a 640x480 canvas at (120, 40)-(520, 440), so the server needs
--crop 120 40 520 440 to see the workspace as the sim camera did.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot.client import ServerError, UnveilerClient
from robot.protocol import decode_image

TESTDATA = Path(__file__).resolve().parent / "testdata"


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    return bool(cond)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jetson-ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--methods", nargs="+", default=["sre", "sre_il", "heuristic", "clip", "random"],
                    help="add gpt4o to exercise the OpenAI baseline (costs an API call)")
    ap.add_argument("--save-dir", default=None, help="write the returned overlays here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    frame = cv2.imread(str(TESTDATA / "frame.png"), cv2.IMREAD_COLOR)
    target = cv2.imread(str(TESTDATA / "target_mask.png"), cv2.IMREAD_GRAYSCALE)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    client = UnveilerClient(args.jetson_ip, args.port)
    ok = True
    try:
        print("ping")
        info = client.ping()
        ok &= check(set(args.methods) <= set(info["methods"]), f"server methods {info['methods']}")
        print(f"        crop={info['crop']} seg_threshold={info['seg_threshold']}")

        print("reset")
        r = client.reset(session="mock", episode=0)
        ok &= check("log_dir" in r, f"log_dir {r.get('log_dir')}")

        for step, method in enumerate(args.methods):
            print(f"select method={method}")
            t0 = time.perf_counter()
            r = client.select(frame, target, method=method, step=step, return_overlay=True)
            rtt = 1e3 * (time.perf_counter() - t0)
            n = r["num_objects"]
            ok &= check(n > 0, f"{n} objects segmented")
            ok &= check(r["target_visible"] and 0 <= r["target_index"] < n, f"target matched to object {r['target_index']}")
            ok &= check(0 <= r["chosen_index"] < n, f"chosen {r['chosen_index']} (is_target={r['is_target']})")
            m = r["chosen_mask_np"]
            ok &= check(m is not None and m.shape == target.shape and m.any(), "chosen_mask at frame size")
            if m is not None and m.any():
                ys, xs = np.nonzero(m)
                cx, cy = r["chosen_centroid"]
                inside = 120 <= xs.min() and xs.max() < 520 and 40 <= ys.min() and ys.max() < 440
                ok &= check(inside and m[cy, cx] > 0, f"mask inside the workspace crop, centroid ({cx}, {cy}) on it")
            if r["probs"] is not None:
                print(f"        probs={r['probs']}")
            print(f"        server timing {r['timing_ms']}  round trip {rtt:.0f} ms")
            if save_dir and r.get("overlay"):
                cv2.imwrite(str(save_dir / f"overlay_{method}.jpg"), decode_image(r["overlay"]))

        print("select with a reachability mask (left half of the workspace only)")
        reach = np.zeros_like(target)
        reach[:, :320] = 255
        r = client.select(frame, target, method="sre", step=90, reachable_mask=reach)
        chosen_ok = r["chosen_index"] < 0 or r["objects"][r["chosen_index"]]["reachable"]
        ok &= check(chosen_ok, f"chosen {r['chosen_index']} is reachable "
                               f"(unfiltered {r['unfiltered_index']}, "
                               f"{sum(o['reachable'] for o in r['objects'])}/{r['num_objects']} reachable)")

        print("select with the target reported hidden (full occlusion)")
        r = client.select(frame, target, method="sre", step=91, target_visible=False)
        ok &= check(not r["target_visible"] and r["target_index"] == -1 and not r["is_target"],
                    f"target hidden, chosen obstacle {r['chosen_index']}")
        r = client.select(frame, target, method="heuristic", step=92, target_visible=False)
        ok &= check(r["chosen_index"] >= 0 and not r["is_target"], f"heuristic, hidden target: chose {r['chosen_index']}")

        print("error path: bad method")
        try:
            client.select(frame, target, method="nope", step=99)
            ok &= check(False, "server should reject an unknown method")
        except ServerError as e:
            ok &= check("unknown method" in str(e), f"ServerError: {e}")
    finally:
        client.close()

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
