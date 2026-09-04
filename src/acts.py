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