import base64
import io
import os
from PIL import Image
from openai import OpenAI

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
        if prompts is None:
            prompts = [
                "an object blocking access to the target",
                "an obstacle preventing grasping the target",
                "the first object that should be removed",
                "an object directly obstructing the target",
                "the most important obstacle to remove"
            ]
    
        # Convert numpy arrays to data URIs
        object_data_uris = [self.np_to_data_uri(obj) for obj in objects]
        target_data_uri = self.np_to_data_uri(target)
        
        # Explicitly index objects
        object_descriptions = "\n".join([f"Object {i}" for i in range(len(objects))])

        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at reasoning about physical scenes. "
                            "Given object crops and a target crop, return ONLY the index "
                            "of the best obstacle to remove. Do not return text."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reasoning prompts:\n" + "\n".join(prompts)},
                        {"type": "text", "text": "Here are the objects in the scene:\n" + object_descriptions},
                        *[
                            {"type": "image_url", "image_url": {"url": uri}}
                            for uri in object_data_uris
                        ],
                        {"type": "text", "text": "Here is the target object:"},
                        {"type": "image_url", "image_url": {"url": target_data_uri}},
                        {"type": "text", "text": "Return only a JSON object with the field `chosen_index` "
                                                "that indicates which object should be removed first."}
                    ]
                }
            ],
            response_format={ 
                "type": "json_schema",
                "json_schema": {
                    "name": "best_obstacle",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "chosen_index": {"type": "integer"}
                        },
                        "required": ["chosen_index"]
                    }
                }
            },
            temperature=0
        )

        print("GPT-4o response:", response.choices[0].message)

        return response.choices[0].message.parsed["chosen_index"]

