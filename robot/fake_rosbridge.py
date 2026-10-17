#!/usr/bin/env python3
"""Minimal stand-in for rosbridge_server, for testing robot.server and robot.client without a Jetson or ROS.

Implements only the rosbridge v2 ops that topic pub/sub over roslibpy uses (advertise, subscribe, publish,
unsubscribe, unadvertise): every `publish` is relayed to all subscribers of that topic. No ROS types, no services.

    python -m robot.fake_rosbridge --port 9090
"""

import argparse
import json
import logging
import sys
from collections import defaultdict

from autobahn.twisted.websocket import WebSocketServerFactory, WebSocketServerProtocol
from twisted.internet import reactor

logger = logging.getLogger("fake_rosbridge")
SUBSCRIBERS = defaultdict(set)   # topic -> {protocol}


class BridgeProtocol(WebSocketServerProtocol):
    def onOpen(self):
        logger.info("client connected: %s", self.peer)

    def onMessage(self, payload, isBinary):
        try:
            msg = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        op, topic = msg.get("op"), msg.get("topic")
        if op == "subscribe":
            SUBSCRIBERS[topic].add(self)
            logger.info("%s subscribed to %s", self.peer, topic)
        elif op == "unsubscribe":
            SUBSCRIBERS[topic].discard(self)
        elif op == "publish":
            out = json.dumps({"op": "publish", "topic": topic, "msg": msg.get("msg", {})}).encode("utf-8")
            for sub in list(SUBSCRIBERS[topic]):
                sub.sendMessage(out, False)
        # advertise / unadvertise / anything else: nothing to do

    def onClose(self, wasClean, code, reason):
        for subs in SUBSCRIBERS.values():
            subs.discard(self)
        logger.info("client disconnected: %s", self.peer)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9090)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    factory = WebSocketServerFactory(f"ws://127.0.0.1:{args.port}")
    factory.protocol = BridgeProtocol
    factory.setProtocolOptions(maxMessagePayloadSize=0, maxFramePayloadSize=0)   # frames are ~100 KB of base64
    reactor.listenTCP(args.port, factory, interface="127.0.0.1")
    logger.info("fake rosbridge on ws://127.0.0.1:%d", args.port)
    reactor.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
