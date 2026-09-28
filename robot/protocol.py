"""Wire format between the Jetson (robot + episode loop) and the lab-PC Unveiler server.

Transport: rosbridge topics of type ``std_msgs/String`` carrying JSON. Images are base64 JPEG, masks are base64 PNG.
The lab PC connects OUT to the Jetson's rosbridge (``ws://<JETSON_IP>:9090``), so nothing listens on the lab PC and
the Jetson opens no extra ports (same pattern as Verify2Act).

    /unveiler/request   Jetson -> lab PC   {"id", "op", ...}
    /unveiler/response  lab PC -> Jetson   {"id", "ok", "error"?, ...}

ops
    ping    {}  -> {methods, default_method, num_patches, sim_image_size, crop, seg_threshold}
    reset   {session, episode}  -> {log_dir}      new episode: start a fresh log folder on the lab PC
    select  {image, target_mask, method?, session?, episode?, step?, reachable_mask?, return_overlay?}
            -> {method, chosen_index, is_target, chosen_reachable, target_visible, target_index, target_reachable,
                num_objects, truncated, objects[{index, centroid, bbox, area, reachable}],
                chosen_centroid, chosen_bbox, chosen_mask, unfiltered_index, probs, reason,
                timing_ms{decode, segment, select, total}, overlay?, log_dir}

``select`` is stateless: every call carries the current camera frame and the target mask, and the server answers
with the single object the chosen method would remove next. The Jetson owns the episode loop, the skill execution and
the success labels.

    image           raw camera frame (BGR, any size), base64 JPEG.
    target_mask     same size as ``image``, nonzero = target, base64 PNG. When the target is hidden, send the mask
                    captured before the occluders were placed; the SRE takes it as its target input either way.
    method          sre (SRE + RL fine-tuning, default) | sre_il (SRE, imitation only) | heuristic | gpt4o | clip | random
    reachable_mask  optional, same size as ``image``, nonzero = the arm can grasp there. Objects whose centroid falls
                    outside are excluded from the choice set for every method (they are still in the scene).

Coordinates in the reply (``centroid`` [x, y], ``bbox`` [x1, y1, x2, y2]) are pixels of the frame the Jetson sent.
``chosen_index`` indexes ``objects``; -1 means nothing can be selected (``reason`` says why). ``is_target`` means the
method wants the target grasped now. ``chosen_mask`` is the chosen object's mask at full frame size (PNG), so the
Jetson can match it against its own colour detections.

Timeouts (Jetson side): ``select`` takes well under a second for sre / sre_il / heuristic / clip / random, and a few
seconds for gpt4o (OpenAI API). ``ping`` / ``reset`` answer immediately.
"""

import base64

import cv2
import numpy as np

REQUEST_TOPIC = "/unveiler/request"
RESPONSE_TOPIC = "/unveiler/response"
MSG_TYPE = "std_msgs/String"

METHODS = ("sre", "sre_il", "heuristic", "gpt4o", "clip", "random")

SELECT_REPLY_TIMEOUT_S = 60.0   # Jetson-side wait for a `select` reply (gpt4o is the slow one)
PING_REPLY_TIMEOUT_S = 5.0


def encode_image(img_bgr: np.ndarray, quality: int = 92) -> str:
    """BGR uint8 image -> base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image(b64: str) -> np.ndarray:
    """base64 JPEG/PNG -> BGR uint8 image."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image payload")
    return img


def encode_mask(mask: np.ndarray) -> str:
    """Binary mask (any dtype, nonzero = on) -> base64 PNG with values 0/255. Lossless, unlike JPEG."""
    ok, buf = cv2.imencode(".png", (np.asarray(mask) > 0).astype(np.uint8) * 255)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_mask(b64: str) -> np.ndarray:
    """base64 PNG -> uint8 mask with values 0/255."""
    m = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("could not decode mask payload")
    return (m > 0).astype(np.uint8) * 255
