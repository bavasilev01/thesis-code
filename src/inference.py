import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from peft import PeftModel
import argparse
import pandas as pd
import os
import numpy as np

from mimic_dataset import MIMICDataset
from prompt import prompt_mimic, prompt_mimic_impression  # the exact same prompt string you used at training time

def run_inference(num_examples: int = 10):
    # ─── 1) Config & device ──────────────────────────────────────────────────
    base_model_path    = "../models/llava-onevision-qwen2-7b-ov-hf"
    lora_adapter_path = "../models/llava-lora-output/checkpoint-1466"
    dataset_csv_path   = "../data/data_1k.csv"

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # ─── 2) Load & merge model ──────────────────────────────────────────────
    print("Loading base OneVision model…")
    processor = AutoProcessor.from_pretrained(base_model_path, use_fast=True)
    processor.tokenizer.padding_side = "right"  # recommended for generation

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    print("Attaching LoRA adapter…")
    peft_model = PeftModel.from_pretrained(model, lora_adapter_path, torch_dtype=torch.float16)
    print("Merging adapter into base model weights…")
    model = peft_model.merge_and_unload()
    model.to(device)
    model.eval()

    # ─── 3) Load dataset ───────────────────────────────────────────────────
    mimic_ds = MIMICDataset(dataset_csv=dataset_csv_path)
    n_avail  = len(mimic_ds)
    print(f"Loaded MIMIC dataset with {n_avail} samples.")
    n = min(num_examples, n_avail)

    # ─── 4) Prepare results storage ──────────────────────────────────────────
    results = []
    
    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)

    # ─── 5) Inference loop ──────────────────────────────────────────────────
    print(f"\nRunning inference on {n} examples…")
    for i in range(n):
        print(f"\n=== Example {i+1}/{n} ===")
        image, true_report, struct_text, chexpert_labels = mimic_ds[i]
        rgb_image = image.convert("RGB")

        # Build conversation template (user part only for inference)
        # This matches the structure used in LLaVAOneVisionDataset where {"type":"image"} is a placeholder
        # and the actual image is passed separately to the processor.
        user_conversation_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Placeholder for the image
                    {"type": "text", "text": prompt_mimic},
                ],
            }
        ]

        # 1. Apply chat template to get the text prompt string
        # add_generation_prompt=True is crucial for generation tasks
        prompt_text = processor.apply_chat_template(
            user_conversation_template,
            add_generation_prompt=True,
            tokenize=False  # Get the string, not tokens
        )

        # 2. Process the text prompt and image using the main processor call
        # This will handle tokenization of text and processing of the image.
        inputs = processor(
            text=[prompt_text],  # processor expects a list of texts
            images=[rgb_image],  # processor expects a list of images
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=1.2,
                do_sample=True,
            )

        # decode full sequence
        full_output = processor.decode(output_ids[0], skip_special_tokens=True).strip()

        # extract just the assistant's reply (after the "assistant" marker)
        marker = "assistant"
        if (marker in full_output):
            predicted = full_output.split(marker, 1)[1].strip()
        else:
            # fallback: drop everything up to the last "\n"
            predicted = full_output.split("\n")[-1].strip()

        print("\n— True Report —")
        print(true_report)
        print("\n— Predicted Report —")
        print(predicted)
        print("="*40)

        # ─── 6) Store results ───────────────────────────────────────────────
        result_entry = {
            'sample_id': i,
            'ground_truth_report': true_report,
            'predicted_report': predicted,
            'structured_text': struct_text,
            'true_chexpert': chexpert_labels.tolist() if isinstance(chexpert_labels, np.ndarray) else chexpert_labels,
            'prompt_used': prompt_mimic,
            'full_model_output': full_output
        }
        
        results.append(result_entry)

    # ─── 7) Save results to CSV ──────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    
    # Save with timestamp for uniqueness
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/inference_results_{timestamp}.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Also save as the standard filename for evaluation
    standard_path = "../results/inference_results.csv"
    results_df.to_csv(standard_path, index=False)
    print(f"Results also saved to: {standard_path}")
    
    # Print summary statistics
    print(f"\nInference Summary:")
    print(f"Total samples processed: {len(results)}")
    print(f"Average predicted report length: {np.mean([len(r['predicted_report']) for r in results]):.1f} characters")
    print(f"Average ground truth report length: {np.mean([len(r['ground_truth_report']) for r in results]):.1f} characters")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=10,
                        help="How many MIMIC samples to run inference on.")
    args = parser.parse_args()
    run_inference(args.num_examples)
