# Basic packages
import random

# Torch and transformers
import torch
from torch.nn.utils.rnn import pad_sequence

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*device_map keys do not match.*")  # Ovis2.5
warnings.filterwarnings("ignore", message=".*slow image processor.*") # Image processor for all models
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*'repr' attribute.*Field().*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*Field().*")



default_response = "[No response generated]"
MAX_TOKENS = 4096

class VLM:
    def __init__(self):
        pass
    def process_input(self):
        raise NotImplementedError
    def infer(self):
        raise NotImplementedError

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _pad_1d(seqs, pad_val):
    return pad_sequence(seqs, batch_first=True, padding_value=pad_val)

def make_collate_fn(ovis_model, mask_user_input=True):
    """
    Collate function that:
      1) runs model.preprocess_inputs on each sample
      2) pads input_ids/labels to the same length
      3) stacks vision tensors (pixel_values, grid_thws) if present
    Notes:
      - Some Ovis builds already implement smart masking internally
        when 'labels' are passed. If yours exposes a helper to build
        labels, prefer that. Here we implement a reasonable default.
    """
    tok = ovis_model.text_tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def collate(batch_messages):
        # Run preprocess per item to keep things simple and robust
        items = []
        for msgs in batch_messages:
            with torch.no_grad():
                # For training we DO NOT add generation prompt
                inputs = ovis_model.preprocess_inputs(
                    messages=msgs,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            items.append(inputs)

        # Extract & pad input_ids
        input_ids_list = [x[0] if isinstance(x, tuple) else x for x in items]  # compat
        input_ids_list = [t.long() for t in input_ids_list]  # [L]
        input_ids = _pad_1d(input_ids_list, pad_id)          # [B, L]
        attention_mask = (input_ids != pad_id).long()

        # Labels: by default train on all tokens.
        labels = input_ids.clone()

        # Optional: mask out everything up to the assistant turn so we only
        # supervise the assistant’s answer. We approximate this by searching
        # for the last BOS/EOS role boundary if the tokenizer uses special tokens.
        # If your remote code exposes better role markers, swap this logic in.
        if mask_user_input:
            # crude heuristic: mask leading tokens until we hit the last occurrence
            # of a conversational boundary token (e.g., </assistant> or <|assistant|>)
            # Fall back silently if not found.
            boundary_tokens = [
                getattr(tok, "assistant_token_id", None),
                getattr(tok, "eos_token_id", None),
            ]
            boundary_tokens = [bt for bt in boundary_tokens if bt is not None]
            for i in range(input_ids.shape[0]):
                ids = input_ids[i]
                cut = None
                for bt in boundary_tokens:
                    pos = (ids == bt).nonzero(as_tuple=False).view(-1)
                    if len(pos) > 0:
                        # start training after the LAST boundary token
                        cut = int(pos[-1].item())
                if cut is not None:
                    labels[i, :cut+1] = -100  # ignore user/source portion
        # Vision tensors
        # items[k] is a tuple like (input_ids, pixel_values, grid_thws)
        pix_list, grid_list = [], []
        has_pix = False
        has_grid = False
        for t in items:
            if isinstance(t, tuple) and len(t) >= 2 and t[1] is not None:
                has_pix = True
                pix_list.append(t[1])
            if isinstance(t, tuple) and len(t) >= 3 and t[2] is not None:
                has_grid = True
                grid_list.append(t[2])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if has_pix:
            # Ovis typically normalizes & resizes to a fixed shape; stack is safe
            batch["pixel_values"] = torch.stack(pix_list, dim=0)
        if has_grid:
            batch["grid_thws"] = torch.stack(grid_list, dim=0)
        return batch
    return collate
