import torch
import wandb
from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig
from transformers import LlavaOnevisionForConditionalGeneration
from transformers import AutoProcessor

from mimic_dataset import MIMICDataset
from llava_dataset import LLaVAOneVisionDataset

# --- Main training script ---
if __name__ == "__main__":
    # 1) Initialize W&B
    wandb.init(project="llava-mimic-finetune", name="llava-lora-run")

    # 2) Load processor, model & apply LoRA
    model_name = "../models/llava-onevision-qwen2-7b-ov-hf"

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = "right"
    tokenizer = processor.tokenizer

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", #"k_proj", 
                        #"dense_h_to_4h",
                        #"dense_4h_to_h",
                        "mm_projector",
                        "cross_attn.q_proj", "cross_attn.v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to("cuda")

    # 3) Prepare dataset & split
    base_ds = MIMICDataset(dataset_csv="../data/data_pa.csv")
    full_ds = LLaVAOneVisionDataset(base_ds, processor, max_length=4096, mode="report")
    train_size = int(0.9 * len(full_ds))
    train_ds, eval_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])

    training_args = TrainingArguments(
        output_dir="../models/llava-lora-output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        learning_rate=1e-5,
        warmup_steps=50,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        remove_unused_columns=False,
        fp16=True,
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds
    )

    trainer.train()
    trainer.save_model("../models/llava-lora-nf4-4h-mm-sharedW-r32-a32-PA-WeightedLoss")
    wandb.finish()
