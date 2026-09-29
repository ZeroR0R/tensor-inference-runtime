# Tensor Inference Runtime

A from scratch rust tensor-inference-runtime, currently supporting Q8 TinyLLama (1.1b) (llama2). Reads a GGUF, parses it, and loads tensors. 

## V1.1

- CPU only (for now)
- GGUF v2/v3 loading
- Memory mapped model
- Llama 2 Forward pass
- Grouped Query Attention (GQA)
- KV Caching (Could be improved)
- Rayon-based parallelism
- Bandwith benchmark
- Golden token tests

No outside runtimes, frameworks, or kernels are used. All readers, tokenizers, and inference operations are implemented in pure Rust.

## Scope

Currently just an experimental CPU inference runtime, not a replacement for llama.cpp.

- Llama 2 architecture
- TinyLlama dimensions
- Q8_0 model weights
- F32 as present

Expansion to other models in the same family is planned. 

## Running

A tinyllama file is used by default:

```text
tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

### Run a prompt:
```bash
cargo run --release -- --prompt "The capital of France is"
```

### Specify a model:
```bash
cargo run --release -- \
  --model ./tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --prompt "The capital of France is"
  ```

### Inspect a model:
```bash
cargo run --release -- --model ./cool_model.gguf --inspect
```

### Forward pass test:
```bash
cargo run --release -- --forward-test
```
