# Running models locally with Ollama

Ollama is a tool for running open large language models on your own computer. It downloads model weights, runs them
efficiently on a CPU or GPU, and exposes a simple HTTP API on port 11434, so applications can talk to a local model without
sending data to a cloud service. This is useful for privacy, offline work and avoiding per-request costs.

Basic commands: `ollama pull llama3.2` downloads a model, `ollama run llama3.2` starts an interactive chat, and `ollama list`
shows the installed models. Applications call the `/api/generate` endpoint for text generation and the `/api/embed` endpoint
to create embeddings. Embedding models such as nomic-embed-text are small and fast, and they are different from chat models.

Smaller models, for example 3 billion parameters, run on modest laptops, while 7 to 8 billion parameter models usually need
around 8 GB of memory or a GPU. Quantisation stores weights with fewer bits, for example 4-bit, which cuts memory use
considerably with a small loss in quality. Lower temperature values make output more deterministic, which is helpful for
question answering over documents.
