import json


message_template = {
    "role": "user",
    "content": [
        {
            "type": "video",
            "video": "",
        },
        {
            "type": "text", 
            "text": ""
        },
    ],
}

def load_eval_data(annotation_path):
    outputs = []
    ground_truths = []
    
    with open(annotation_path, "r") as f:
        examples = json.load(f)

    for example in examples:
        message = message_template.copy()
        message["content"][0]["video"] = example["videos"][0]
        message["content"][1]["text"] = example["conversations"][0]["value"]
        ground_truth = example["conversations"][1]["value"]
        outputs.append(message)
        ground_truths.append(ground_truth)

    return outputs, ground_truths