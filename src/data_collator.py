import torch
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class LLaVADataCollator:
    """
    Data collator for LLaVA training that handles batching of variable-length sequences
    and image data properly.
    """
    tokenizer: Any
    padding: bool = True
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        
        # Handle text data (input_ids, attention_mask, labels)
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad sequences to the same length within the batch
        max_len = max(len(ids) for ids in input_ids)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        
        # Pad input_ids
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(input_ids)):
            ids = input_ids[i][:max_len]  # Truncate if needed
            attn = attention_mask[i][:max_len]
            lbls = labels[i][:max_len]
            
            # Pad to max_len
            pad_length = max_len - len(ids)
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id
                ids = torch.cat([ids, torch.full((pad_length,), pad_token_id, dtype=ids.dtype)])
                attn = torch.cat([attn, torch.zeros(pad_length, dtype=attn.dtype)])
                lbls = torch.cat([lbls, torch.full((pad_length,), -100, dtype=lbls.dtype)])
            
            padded_input_ids.append(ids)
            padded_attention_mask.append(attn)
            padded_labels.append(lbls)
        
        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(padded_attention_mask)
        batch["labels"] = torch.stack(padded_labels)
        
        # Handle image data
        pixel_values = [f["pixel_values"] for f in features]
        batch["pixel_values"] = torch.stack(pixel_values)
        
        # Handle image sizes
        image_sizes = [f["image_sizes"] for f in features]
        batch["image_sizes"] = torch.stack(image_sizes)
        
        # Handle chexpert labels
        chexpert_labels = [f["chexpert_labels"] for f in features]
        batch["chexpert_labels"] = torch.stack(chexpert_labels)
        
        return batch
