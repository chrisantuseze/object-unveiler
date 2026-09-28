#!/usr/bin/env python3
"""Unveiler object-selection server for the real DOFBOT. Runs on the lab GPU PC, not on the Jetson.

It connects OUT to the Jetson's rosbridge with roslibpy (no ROS install needed here), listens on
/unveiler/request and answers on /unveiler/response (see protocol.py). The Jetson keeps the episode loop and skill
execution; this process only segments the frame and picks the next object to remove.

    Jetson:  roscore, arm_driver, camera, ..., roslaunch rosbridge_server rosbridge_websocket.launch
    Lab PC:  conda activate unveiler
             python -m robot.server --jetson-ip 192.168.0.8 [--crop X0 Y0 X1 Y1]

Offline check without the Jetson (loads the models, selects once on image files):
    python -m robot.server --offline-image frame.png --offline-target target_mask.png --method sre
"""

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot.protocol import MSG_TYPE, REQUEST_TOPIC, RESPONSE_TOPIC, decode_image, decode_mask

logging.getLogger("twisted").setLevel(logging.WARNING)
logger = logging.getLogger("unveiler_server")


class Server:
    def __init__(self, backend, jetson_ip: str, port: int = 9090, connect_timeout: float = 15.0):
        import roslibpy

        self._roslibpy = roslibpy
        self.backend = backend
        self.lock = threading.Lock()   # one model call at a time (single GPU)

        logger.info("Connecting to rosbridge ws://%s:%d ...", jetson_ip, port)
        self.ros = roslibpy.Ros(host=jetson_ip, port=port)
        try:
            self.ros.run(timeout=connect_timeout)
        except Exception:
            pass   # roslibpy raises on timeout; report it below with the actionable message
        if not self.ros.is_connected:
            raise ConnectionError(f"Could not reach rosbridge at {jetson_ip}:{port}. Is "
                                  "`roslaunch rosbridge_server rosbridge_websocket.launch` running on the Jetson?")
        self.pub = roslibpy.Topic(self.ros, RESPONSE_TOPIC, MSG_TYPE)
        self.pub.advertise()
        self.sub = roslibpy.Topic(self.ros, REQUEST_TOPIC, MSG_TYPE)
        self.sub.subscribe(self.on_request)
        logger.info("Ready: listening on %s, answering on %s", REQUEST_TOPIC, RESPONSE_TOPIC)

    def on_request(self, msg: dict):
        # roslibpy callbacks run on the websocket thread; work elsewhere so pings stay responsive.
        threading.Thread(target=self.handle, args=(msg["data"],), daemon=True).start()

    def handle(self, raw: str):
        reply = handle_request(self.backend, raw, self.lock)
        if reply is not None:
            self.pub.publish(self._roslibpy.Message({"data": json.dumps(reply)}))

    def run(self):
        try:
            while self.ros.is_connected:
                time.sleep(1.0)
            logger.error("Lost the rosbridge connection.")
        except KeyboardInterrupt:
            pass
        finally:
            for fn in (self.sub.unsubscribe, self.pub.unadvertise, self.ros.terminate):
                try:
                    fn()
                except Exception:
                    pass


def handle_request(backend, raw: str, lock: threading.Lock):
    """Decode one request, run it, and return the reply dict (None if the request is not JSON)."""
    try:
        req = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring non-JSON request: %r", raw[:200])
        return None
    rid, op = req.get("id"), req.get("op")
    t0 = time.time()
    try:
        if op == "ping":
            out = dict(backend.settings)   # never takes the model lock
        else:
            with lock:
                out = dispatch(backend, op, req)
        reply = {"id": rid, "ok": True, **out}
    except Exception as e:
        logger.exception("op %s failed", op)
        reply = {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}
    extra = ""
    if op == "select" and reply["ok"]:
        extra = (f"method={reply['method']} chosen={reply['chosen_index']} target={reply['target_index']} "
                 f"is_target={reply['is_target']} n={reply['num_objects']} {reply['reason']}")
    logger.info("%-6s %.2fs %s", op, time.time() - t0, extra)
    return reply


def dispatch(backend, op: str, r: dict) -> dict:
    if op == "reset":
        return backend.reset(r.get("session", "default"), r.get("episode", 0))
    if op == "select":
        t0 = time.perf_counter()
        image = decode_image(r["image"])
        target = decode_mask(r["target_mask"])
        reachable = decode_mask(r["reachable_mask"]) if r.get("reachable_mask") else None
        decode_ms = 1e3 * (time.perf_counter() - t0)
        return backend.select(image, target, method=r.get("method", "sre"), reachable_mask=reachable,
                              target_visible=r.get("target_visible"), step=r.get("step"),
                              return_overlay=bool(r.get("return_overlay")), decode_ms=decode_ms)
    raise ValueError(f"unknown op '{op}'")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Unveiler real-robot object-selection server (lab PC side)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--jetson-ip", default="192.168.0.8")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sre-rl-ckpt", default="save/sre_rl/sre_rl_best.pt")
    ap.add_argument("--sre-il-ckpt", default="save/sre/sre_model_best.pt")
    ap.add_argument("--num-patches", type=int, default=10, help="SRE object slots (must match training)")
    ap.add_argument("--seg-threshold", type=float, default=0.97,
                    help="Mask R-CNN score threshold (ObjectSegmenter uses 0.97 for real, 0.98 for sim)")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None,
                    help="workspace ROI in camera pixels; the heuristic treats the ROI border as the workspace edge, "
                         "as the sim camera did. Default: whole frame")
    ap.add_argument("--output-dir", default="save/real_eval", help="per-session step logs")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random selector")

    ap.add_argument("--offline-image", default=None, help="skip rosbridge: select once on this image and exit")
    ap.add_argument("--offline-target", default=None, help="target mask for --offline-image (nonzero = target)")
    ap.add_argument("--method", default="sre", help="method for --offline-image")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from robot.backend import UnveilerBackend
    backend = UnveilerBackend.from_args(args)

    if args.offline_image:
        import cv2
        image = cv2.imread(args.offline_image, cv2.IMREAD_COLOR)
        target = cv2.imread(args.offline_target, cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            raise SystemExit("could not read --offline-image / --offline-target")
        backend.reset("offline", 0)
        reply = backend.select(image, target, method=args.method, step=0)
        print(json.dumps({k: v for k, v in reply.items() if k != "chosen_mask"}, indent=2))
        return 0

    Server(backend, args.jetson_ip, args.port).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
