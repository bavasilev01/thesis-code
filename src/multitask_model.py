import torch
import torch.nn as nn
from transformers import LlavaOnevisionForConditionalGeneration
from transformers import AutoProcessor, BitsAndBytesConfig

from mimic_dataset import MIMICDataset
from llava_dataset import LLaVAOneVisionDataset

class MultiTaskLlavaOnevisionForConditionalGeneration(nn.Module):
    """
    Wraps a pre-trained LlavaOnevisionForConditionalGeneration model to add a
    multi-label CheXpert classification head on top of the encoder's pooled output,
    combining BCE loss on CheXpert labels with the generation LM loss.
    """
    def __init__(
        self,
        base_model: LlavaOnevisionForConditionalGeneration,
        num_labels: int = 14,
        hidden_size: int = 3584,
        lambda_weight: float = 0.5
    ):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.lambda_weight = lambda_weight

        # TODO: This is MLP, introduce a hidden layer
        self.classifier = nn.Linear(hidden_size, num_labels)
        # BCE loss; we'll mask out uncertain labels (value -1)
        self.bce_fn = nn.BCEWithLogitsLoss(reduction="none")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_labels: int = 14,
        lambda_weight: float = 0.5,
        **kwargs
    ):
        """
        Load the base LLaVA model via from_pretrained(), then wrap it.
        """
        base = LlavaOnevisionForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        return cls(base, num_labels=num_labels, lambda_weight=lambda_weight)

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        image_sizes=None,
        labels=None,
        chexpert_labels=None,
        **kwargs
    ):
        # 1) Forward through the base model (auto-regressive LM)
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_sizes=image_sizes,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        lm_loss = outputs.loss
        hidden_states = outputs.hidden_states  # tuple of (B, L, H)

        # 2) Mean-pool the last hidden states over non-padded tokens
        last_h = hidden_states[-1]  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_h * mask).sum(dim=1)           # (B, H)
        counts = mask.sum(dim=1).clamp(min=1)         # (B, 1)
        pooled = summed / counts                      # (B, H)

        # 3) Compute CheXpert logits
        chex_logits = self.classifier(pooled)        # (B, num_labels)

        # 4) If provided, compute BCE loss on chexpert_labels (masking -1)
        if chexpert_labels is not None:
            valid = (chexpert_labels != -1).float()    # (B, num_labels)
            targets = chexpert_labels.clamp(min=0)     # treat -1 as ignored
            per_label_loss = self.bce_fn(chex_logits, targets) * valid
            denom = valid.sum(dim=1).clamp(min=1)      # (B,)
            chex_loss = (per_label_loss.sum(dim=1) / denom).mean()

            # 5) Combine LM loss + weighted chexpert loss
            total_loss = lm_loss + self.lambda_weight * chex_loss
            outputs.loss = total_loss
            outputs.lm_loss = lm_loss
            outputs.chexpert_loss = chex_loss

        # 6) Always expose chexpert logits
        outputs.chexpert_logits = chex_logits
        return outputs

# Quick sanity check
if __name__ == "__main__":
    model_name = "../models/llava-onevision-qwen2-7b-ov-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = "right"
    tokenizer = processor.tokenizer

    base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = MultiTaskLlavaOnevisionForConditionalGeneration(
        base_model=base_model,
        num_labels=14,
        hidden_size=3584,
        lambda_weight=0.5
    )
    model.to("cuda")
    
    # Dataset loading
    base_ds = MIMICDataset(dataset_csv="../data/data_pa.csv")
    full_ds = LLaVAOneVisionDataset(base_ds, processor, max_length=3584, mode="report")

    # Forward pass
    batch = full_ds[0]
    labels_for_decoding = batch['labels'].clone()
    labels_for_decoding[labels_for_decoding == -100] = tokenizer.pad_token_id
    label_text = tokenizer.decode(labels_for_decoding, skip_special_tokens=True)
    print("True report text:", label_text)
    # Unsqueeze tensor inputs
    for k in ("input_ids","attention_mask","labels","pixel_values","chexpert_labels"):
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].unsqueeze(0)
    batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}

    # 4) Forward
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        image_sizes=batch["image_sizes"],
        labels=batch["labels"],
        chexpert_labels=batch["chexpert_labels"],
        use_cache=False,
    )

    pred_text = tokenizer.decode(outputs.logits[0].argmax(dim=-1), skip_special_tokens=True)

    print("Generated text:", pred_text[:100])
    print("CheXpert logits:", outputs.chexpert_logits)
    print("CheXpert labels:", batch["chexpert_labels"])
    print("Total loss:", outputs.loss.item())
