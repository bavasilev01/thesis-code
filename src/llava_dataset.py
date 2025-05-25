import torch
from torch.utils.data import Dataset
from prompt import prompt_mimic, prompt_mimic_impression

class LLaVAOneVisionDataset(Dataset):
    def __init__(self, mimic_dataset, processor, max_length=1024, mode='report'):
        """
        mimic_dataset: returns (PIL.Image, report_text, structured_text, chexpert_labels)
        processor: AutoProcessor or LlavaOnevisionProcessor
        """
        self.dataset     = mimic_dataset
        self.processor   = processor
        self.tokenizer   = processor.tokenizer
        self.max_length  = max_length
        self.mode        = mode
        # ensure padding on the right
        self.tokenizer.padding_side = "right"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1) load raw data
        image, report_text, structured_text, chexpert_labels = self.dataset[idx]
        target_text = report_text if self.mode == 'report' else structured_text

        # 2) build chat prompts
        conv = [
            {"role":"user","content":[
               {"type":"image"},
               {"type":"text","text":prompt_mimic}
            ]},
            {"role":"assistant","content":[{"type":"text","text":target_text}]}
        ]
        # record the original image size so the model can token‐patch correctly
        image_sizes = torch.tensor([[image.height, image.width]], dtype=torch.long)

        # 1) full‐conv encoding (user + target + gen‐prompt)
        enc_full = self.processor(
            text=[ self.processor.apply_chat_template(conv, add_generation_prompt=True) ],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids      = enc_full.input_ids.squeeze(0)        # [L]
        attention_full = enc_full.attention_mask.squeeze(0)   # [L]
        pixel_values   = enc_full.pixel_values.squeeze(0)     # [3,H,W]

        # 2) user‐only encoding (just the user turn + gen‐prompt)
        user_conv = [ conv[0] ]   # keep only {"role":"user",...}
        enc_user = self.processor(
            text=[ self.processor.apply_chat_template(user_conv, add_generation_prompt=True) ],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        attention_user = enc_user.attention_mask.squeeze(0)   # [L]

        # 3) build labels by masking out everything the user saw
        labels = input_ids.clone()
        # 3a) mask prefix (all positions where attention_user==1)
        labels[attention_user == 1] = -100
        # 3b) mask padding in the full conv
        labels[attention_full == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_full,
            "pixel_values":   pixel_values,
            "image_sizes":    image_sizes,
            "labels":         labels,
            "chexpert_labels": chexpert_labels
        }

if __name__ == "__main__":
    # Example usage
    from mimic_dataset import MIMICDataset
    from transformers import LlavaOnevisionProcessor

    processor = LlavaOnevisionProcessor.from_pretrained("../models/llava-onevision-qwen2-7b-ov-hf")
    mimic_dataset = MIMICDataset(dataset_csv="../data/data_pa.csv")
    dataset = LLaVAOneVisionDataset(mimic_dataset, processor, max_length=4096, mode='report')
    tokenizer = processor.tokenizer

    # Test the dataset
    for i in range(1): # Check only the first sample for brevity
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"Pixel Values Shape: {sample['pixel_values'].shape}")
        print(f"Image Sizes: {sample['image_sizes']}")
        print("Chexpert_labels:", sample["chexpert_labels"].shape, sample["chexpert_labels"])

        # Decode input_ids
        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"\nDecoded Input Text:\n{input_text}")

        # Decode labels (replace -100 with pad_token_id for decoding)
        labels_for_decoding = sample['labels'].clone()
        labels_for_decoding[labels_for_decoding == -100] = tokenizer.pad_token_id
        label_text = tokenizer.decode(labels_for_decoding, skip_special_tokens=True)
        print(f"\nDecoded Label Text:\n{label_text}")

