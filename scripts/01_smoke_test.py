import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

if torch.cuda.is_available():
    DEVICE, DTYPE = "cuda", torch.float16
elif torch.backends.mps.is_available():
    DEVICE, DTYPE = "mps", torch.float32
else:
    DEVICE, DTYPE = "cpu", torch.float32

print(f"device={DEVICE} dtype={DTYPE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE
)
model.eval()

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

cfg = model.config
N_LAYERS = cfg.num_hidden_layers
D_MODEL = cfg.hidden_size
print(f"n_layers={N_LAYERS}  d_model={D_MODEL}  vocab={cfg.vocab_size}")


# Check shapes
def chat(prompt: str) -> str:
    """Wrap a bare string in the model's chat template.

    Skipping this is the second most common quiet bug. A base-format prompt sent
    to an instruct-tuned model puts you off-distribution, and your probe may end
    up reading 'this text looks malformed' rather than the concept you wanted.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


prompt = chat("The city of Paris is in France. Is that true or false?")
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

hs = out.hidden_states

assert len(hs) == N_LAYERS + 1, f"expected {N_LAYERS + 1} hidden states, got {len(hs)}"
assert hs[0].shape == (1, inputs.input_ids.shape[1], D_MODEL), hs[0].shape
print(f"hidden_states: tuple of {len(hs)}, each {tuple(hs[0].shape)} = (batch, seq, d_model)")


with torch.no_grad():
    direct = model.lm_head(hs[-1])

matches = torch.allclose(direct, out.logits, atol=1e-2)
print(f"lm_head(hidden_states[-1]) == logits: {matches}")
print("  -> final norm IS already applied" if matches else
      "  -> final norm is NOT applied; call model.model.norm() before unembedding")


# Forward hooks
captured = {}


def make_hook(name):
    def hook(module, layer_input, layer_output):
        # Decoder blocks return a tuple, the residual stream is element 0.
        captured[name] = (layer_output[0] if isinstance(layer_output, tuple)
                          else layer_output).detach()
    return hook


LAYER = N_LAYERS // 2
handle = model.model.layers[LAYER].register_forward_hook(make_hook("mid"))
with torch.no_grad():
    model(**inputs)
handle.remove()  # ALWAYS remove. Leaked hooks stack silently and corrupt later runs.

# The hook fires on the output of block LAYER, which is hs[LAYER + 1].
assert torch.allclose(captured["mid"], hs[LAYER + 1], atol=1e-3), \
    "hook output and hidden_states disagree"
print(f"hook on layers[{LAYER}] matches hidden_states[{LAYER + 1}]")


prompts = [
    chat("Is the sky blue?"),
    chat("Is the following statement true or false: the Nile is in South America?"),
    chat("Hi."),
]

batch = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    batched = model(**batch, output_hidden_states=True).hidden_states[LAYER + 1]

# With left padding every sequence's last real token sits at index -1.
batched_last = batched[:, -1, :]

solo_last = []
for p in prompts:
    ii = tokenizer(p, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        solo_last.append(model(**ii, output_hidden_states=True).hidden_states[LAYER + 1][0, -1, :])
solo_last = torch.stack(solo_last)

max_diff = (batched_last - solo_last).abs().max().item()
print(f"batched vs unbatched last-token, max abs diff: {max_diff:.4f}")
assert max_diff < 0.5, "batching is changing your activations -- check padding_side"

true_stmts = ["Paris is the capital of France.", "Water is composed of hydrogen and oxygen."]
false_stmts = ["Paris is the capital of Japan.", "Water is composed of iron and neon."]


def last_token_acts(texts, layer):
    b = tokenizer([chat(t) for t in texts], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        return model(**b, output_hidden_states=True).hidden_states[layer + 1][:, -1, :].float()


print("\nlayer | within-class sim | across-class sim | gap")
for layer in range(0, N_LAYERS, max(1, N_LAYERS // 8)):
    t = torch.nn.functional.normalize(last_token_acts(true_stmts, layer), dim=-1)
    f = torch.nn.functional.normalize(last_token_acts(false_stmts, layer), dim=-1)
    within = ((t[0] @ t[1]) + (f[0] @ f[1])).item() / 2
    across = (t @ f.T).mean().item()
    print(f"{layer:5d} | {within:16.3f} | {across:16.3f} | {within - across:+.3f}")

print("\nA positive gap that grows through the middle layers is the expected shape.")
print("Four examples is far too few to conclude anything. This is a smoke test.")

b = tokenizer([chat(s) for s in true_stmts + false_stmts], return_tensors="pt", padding=True)
print(repr(tokenizer.decode(b.input_ids[0, -1])))
print([repr(tokenizer.decode(row[-1])) for row in b.input_ids])