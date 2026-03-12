import os
import re
import json
import argparse

from qwenvl.dataset.eval_data import load_eval_data
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

def parse_output(text: str):
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.S)
    think_content = think_match.group(1).strip() if think_match else ""

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    answer_text = answer_match.group(1).strip() if answer_match else ""

    expected_keys = [
        "Video Quality",
        "Subject Movement",
        "Physical Interaction",
        "Cause-Effect",
        "Subject Existence",
        "Object Existence",
        "Subject-Object Interaction",
    ]

    pairs = re.findall(r"([\w\- ]+):\s*([^.\n]+)\.?", answer_text)
    raw_dict = {k.strip(): v.strip() for k, v in pairs}
    answer_dict = {k: raw_dict.get(k, "Fail") for k in expected_keys}

    return think_content, answer_dict

def calculate_socre(outputs):
    score = 0
    cnt = 0
    for output in outputs:
        answer = output["answer"]
        gt = output["ground_truth"]
        
        _, answer_dict = parse_output(answer)
        _, gt_answer_dict = parse_output(gt)

        for key in answer_dict.keys():
            if answer_dict[key].lower().replace("good", "yes").replace("bad", "no") == gt_answer_dict[key].lower():
                score += 1
            else:
                score += 0
            cnt += 1
    
    return score / cnt

def main(args): 
    model_name = args.model_path.split("/")[1]
    output_path = os.path.join("eval_results", model_name + ".json")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            outputs = json.load(f)
        score = calculate_socre(outputs)
        print(f"Score: {score}")
        return None

    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto"
    )

    try:
        processor = AutoProcessor.from_pretrained(args.model_path)
    except:
        processor = AutoProcessor.from_pretrained(os.path.dirname(args.model_path))

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
                "video_path": data["content"][0]["video"],
                "answer": output_text[0],
                "ground_truth": gt,   
            }
        )

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)

    score = calculate_socre(outputs)
    print(f"Score: {score}")
    return None

if __name__ == "__main__":
    main(args)