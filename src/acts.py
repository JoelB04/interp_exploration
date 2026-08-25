"""
Activation extraction. Put this at src/acts.py.

Split out from the smoke test because session 2 is the second time we need it,
and that is the right moment to make something a module -- not before.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load(model_name: str = MODEL_NAME):
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)
    model.eval()

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok, device


def format_prompts(statements, tok, mode: str):
    """Two readout regimes.

    'raw'  -- the bare statement. The last token is the final token of the
              statement itself (usually the period). The concept is read where
              it was computed.

    'chat' -- statement wrapped in the chat template. The last token is the
              generation-prompt token, which is IDENTICAL across all examples.
              Any signal there had to be moved by attention from the statement.

    This is the variable you discovered in session 1. Do not hardcode it.
    """
    if mode == "raw":
        return list(statements)
    if mode == "chat":
        return [
            tok.apply_chat_template(
                [{"role": "user", "content": f"Is the following true or false?\n{s}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for s in statements
        ]
    raise ValueError(f"unknown mode: {mode}")


@torch.no_grad()
def get_acts(statements, model, tok, device, mode="raw", batch_size=16):
    """Last-token residual stream at every layer.

    Returns a float32 tensor of shape (n_examples, n_layers + 1, d_model).

    That middle axis is n_layers + 1, and index i is the output of block i - 1.
    Index 0 is the embedding. Keep the off-by-one in the tensor rather than
    silently dropping it -- you want the embedding layer as a baseline.
    """
    prompts = format_prompts(statements, tok, mode)
    out = []

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        # Left padding means index -1 is the last real token for every row.
        stacked = torch.stack([h[:, -1, :] for h in hs], dim=1)  # (batch, layers+1, d)
        out.append(stacked.float().cpu())

        done = min(i + batch_size, len(prompts))
        print(f"\r  activations {done}/{len(prompts)}", end="", flush=True)

    print()
    return torch.cat(out, dim=0)