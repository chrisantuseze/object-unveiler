import base64
import io
import os
from PIL import Image
from openai import OpenAI
import json

class GPTRemovalPredictor:
    def __init__(self):
        api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=api_key)

    def np_to_data_uri(self, np_img, format="PNG"):
        """Convert a numpy array (H,W,C) into a base64 data URI string."""
        pil_img = Image.fromarray(np_img.astype("uint8"))
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"

    def predict(self, objects, target, prompts=None):
        """
        objects: list of numpy arrays (object crops)
        target: numpy array (target crop)
        returns: int (index of best obstacle to remove)
        """
        if prompts is None:
            prompts = [
                "an object blocking access to the target",
                "an obstacle preventing grasping the target",
                "the first object that should be removed",
                "an object directly obstructing the target",
                "the most important obstacle to remove"
            ]

        # Convert numpy arrays to base64 data URIs
        object_data_uris = [self.np_to_data_uri(obj) for obj in objects]
        target_data_uri = self.np_to_data_uri(target)

        # Build object-indexed content
        object_contents = []
        for i, uri in enumerate(object_data_uris):
            object_contents.append({"type": "text", "text": f"Object {i}:"})
            object_contents.append({"type": "image_url", "image_url": {"url": uri}})

        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are evaluating a robotics task. "
                        "Given several objects and a target, identify which object "
                        "is the most important obstacle preventing access to the target. "
                        "Rules:\n"
                        "- Return ONLY the index of the chosen object.\n"
                        "- Never choose the target itself unless the target is not obstructed and can be grasped.\n"
                        "- If there are only two objects, always return the target.\n"
                        "- Think step by step internally, but output ONLY JSON.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reasoning prompts:\n" + "\n".join(prompts)},
                        {"type": "text", "text": "Here are the objects in the scene:"},
                        *object_contents,
                        {"type": "text", "text": "Target object:"},
                        {"type": "image_url", "image_url": {"url": target_data_uri}},
                        {"type": "text", "text": "Return only a JSON object with the field `chosen_index`."},
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "best_obstacle",
                    "schema": {
                        "type": "object",
                        "properties": {"chosen_index": {"type": "integer"}},
                        "required": ["chosen_index"],
                    },
                },
            },
            temperature=0,
        )

        chosen_index = json.loads(response.choices[0].message.content)["chosen_index"]
        print("GPT-4o response:", chosen_index)

        return chosen_index

