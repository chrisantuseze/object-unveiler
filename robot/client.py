"""Jetson-side client for the Unveiler server: one request/response channel over rosbridge.

Needs only roslibpy, numpy and OpenCV (no torch, no ROS Python packages), so it can be copied as-is into
dofbot-controller next to protocol.py. Usage:

    from protocol import encode_image, encode_mask
    client = UnveilerClient("127.0.0.1")        # on the Jetson, rosbridge is local
    client.ping()                               # raises TimeoutError if the lab-PC server is not running
    client.reset(session="sre_d1_partial", episode=3)
    reply = client.select(frame_bgr, target_mask, method="sre", step=0)
    client.close()
"""

import json
import logging
import threading
import uuid
from typing import Optional

import numpy as np

try:   # inside object-unveiler
    from robot.protocol import (MSG_TYPE, PING_REPLY_TIMEOUT_S, REQUEST_TOPIC, RESPONSE_TOPIC,
                                SELECT_REPLY_TIMEOUT_S, decode_mask, encode_image, encode_mask)
except ImportError:
    try:   # copied into dofbot-controller/unveiler/
        from unveiler.protocol import (MSG_TYPE, PING_REPLY_TIMEOUT_S, REQUEST_TOPIC, RESPONSE_TOPIC,
                                       SELECT_REPLY_TIMEOUT_S, decode_mask, encode_image, encode_mask)
    except ImportError:   # copied next to protocol.py, run from that folder
        from protocol import (MSG_TYPE, PING_REPLY_TIMEOUT_S, REQUEST_TOPIC, RESPONSE_TOPIC,
                              SELECT_REPLY_TIMEOUT_S, decode_mask, encode_image, encode_mask)

logger = logging.getLogger(__name__)


class ServerError(RuntimeError):
    """The server answered with ok=false."""


class UnveilerClient:
    def __init__(self, jetson_ip: str = "127.0.0.1", port: int = 9090, connect_timeout: float = 15.0, ros=None):
        import roslibpy

        self._roslibpy = roslibpy
        self._own_ros = ros is None
        if ros is None:
            ros = roslibpy.Ros(host=jetson_ip, port=port)
            try:
                ros.run(timeout=connect_timeout)
            except Exception:
                pass
            if not ros.is_connected:
                raise ConnectionError(f"Could not reach rosbridge at {jetson_ip}:{port}")
        self.ros = ros
        self._pending = {}
        self._lock = threading.Lock()
        self.pub = roslibpy.Topic(ros, REQUEST_TOPIC, MSG_TYPE)
        self.pub.advertise()
        self.sub = roslibpy.Topic(ros, RESPONSE_TOPIC, MSG_TYPE)
        self.sub.subscribe(self._on_response)

    def _on_response(self, msg: dict):
        try:
            reply = json.loads(msg["data"])
        except (KeyError, json.JSONDecodeError):
            return
        with self._lock:
            slot = self._pending.get(reply.get("id"))
        if slot is not None:   # replies to other clients' ids are ignored
            slot["reply"] = reply
            slot["event"].set()

    def call(self, op: str, timeout: float, **fields) -> dict:
        rid = uuid.uuid4().hex
        slot = {"event": threading.Event(), "reply": None}
        with self._lock:
            self._pending[rid] = slot
        try:
            self.pub.publish(self._roslibpy.Message({"data": json.dumps({"id": rid, "op": op, **fields})}))
            if not slot["event"].wait(timeout):
                raise TimeoutError(f"no reply to '{op}' within {timeout:.0f}s (is robot.server running on the lab PC?)")
        finally:
            with self._lock:
                self._pending.pop(rid, None)
        reply = slot["reply"]
        if not reply.get("ok"):
            raise ServerError(reply.get("error", "unknown server error"))
        return reply

    def ping(self, timeout: float = PING_REPLY_TIMEOUT_S) -> dict:
        return self.call("ping", timeout)

    def reset(self, session: str, episode: int) -> dict:
        return self.call("reset", PING_REPLY_TIMEOUT_S, session=session, episode=int(episode))

    def select(self, frame_bgr: np.ndarray, target_mask: np.ndarray, method: str = "sre",
               step: Optional[int] = None, reachable_mask: Optional[np.ndarray] = None,
               target_visible: Optional[bool] = None, return_overlay: bool = False,
               timeout: float = SELECT_REPLY_TIMEOUT_S) -> dict:
        """Ask which object to remove next. Adds reply['chosen_mask_np'] (uint8 0/255, frame size) when one is chosen."""
        fields = {"image": encode_image(frame_bgr), "target_mask": encode_mask(target_mask), "method": method,
                  "return_overlay": return_overlay}
        if step is not None:
            fields["step"] = int(step)
        if reachable_mask is not None:
            fields["reachable_mask"] = encode_mask(reachable_mask)
        if target_visible is not None:
            fields["target_visible"] = bool(target_visible)
        reply = self.call("select", timeout, **fields)
        reply["chosen_mask_np"] = decode_mask(reply["chosen_mask"]) if reply.get("chosen_mask") else None
        return reply

    def close(self):
        for fn in (self.sub.unsubscribe, self.pub.unadvertise):
            try:
                fn()
            except Exception:
                pass
        if self._own_ros:
            try:
                self.ros.terminate()
            except Exception:
                pass
