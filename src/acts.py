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
    """Three readout regimes.

    'raw'     -- the bare statement. The last token is the final token of the
                 statement itself. The concept is read where it was computed.

    'chat'    -- wrapped in the chat template AND in a true/false question.
                 The last token is the generation-prompt token, identical
                 across all examples, so any signal there was moved by
                 attention from the statement.

                 CAUTION, found 2026-08-28. The "Is the following true or
                 false?" framing is a leftover from the closed truth-probe
                 project (git 73f06af). For SALAD prompts it means the model is
                 asked to JUDGE THE TRUTH of a harmful request rather than to
                 respond to it -- its top next tokens are 'True' and 'False'.
                 Every 'chat' result on SALAD measures that framing, not a
                 request. Kept unchanged so existing cached activations remain
                 correctly labelled; use 'request' for new work.

    'request' -- the statement as a plain user turn, generation prompt
                 appended, no task framing. What you would actually send to a
                 chat model. Use this for anything about how the model responds
                 to the prompt.

    Readout position is a live experimental variable in this repo. Never
    hardcode it, and never change what an existing mode name means -- the
    activation cache keys on the mode string, so redefining one in place would
    let stale tensors be silently reused under a new meaning.
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
    if mode == "request":
        return [
            tok.apply_chat_template(
                [{"role": "user", "content": s}],
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