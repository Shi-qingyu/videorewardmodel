import json

from qwenvl.dataset.eval_data import load_eval_data
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "output/qwen3vl-2b-baseline/checkpoint-158", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("output/qwen3vl-2b-baseline")

dataset, ground_truths = load_eval_data("./data/eval.json")

outputs = []
for data, gt in zip(dataset, ground_truths):
    inputs = [data]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        inputs,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    outputs.append(
        {
            "video_path": inputs[0]["content"][0]["video"],
            "answer": output_text[0],
            "ground_truth": gt,   
        }
    )

with open("eval_results.json", "w") as f:
    json.dump(outputs, f, indent=4)