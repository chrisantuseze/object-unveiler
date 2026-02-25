# inference_server.py  — runs on lab computer (192.168.0.113)
import roslibpy
import numpy as np
import time

# --- Load your ML models here ---
# from policy.models import YourModel
# model = YourModel.load("path/to/weights")

JETSON_IP = "192.168.0.8"
JETSON_PORT = 9090

client = roslibpy.Ros(host=JETSON_IP, port=JETSON_PORT)
client.run()

print(f"Connected to Jetson: {client.is_connected}")

# ── Subscribe to a sensor topic ──────────────────────────────────────────────
action_publisher = roslibpy.Topic(client, '/robot/action_cmd', 'std_msgs/Float32MultiArray')

def on_observation(msg):
    """Called whenever the Jetson publishes an observation."""
    # Decode the ROS message into a numpy array
    obs = np.array(msg['data'], dtype=np.float32)

    # ── Run your ML model ──
    # action = model.predict(obs)
    action = obs * 0.0  # placeholder

    # ── Publish the action back to the Jetson ──
    action_msg = roslibpy.Message({'data': action.tolist()})
    action_publisher.publish(action_msg)
    print(f"Sent action: {action}")

obs_subscriber = roslibpy.Topic(client, '/robot/observation', 'std_msgs/Float32MultiArray')
obs_subscriber.subscribe(on_observation)

try:
    while client.is_connected:
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    obs_subscriber.unsubscribe()
    client.terminate()