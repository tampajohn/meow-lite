# meow-lite

A "real" LLM API shim: an OpenAI- and Anthropic-compatible HTTP server that accepts
your chat completions and messages requests, thinks very hard about them, and responds
exactly as a cat would — in meows. Drop-in compatible enough to point any SDK at it;
technically state-of-the-art in feline alignment.

## Quickstart

```bash
/opt/homebrew/bin/uv run uvicorn meow_lite.server:app --port 8011
```

## Examples

OpenAI-compatible chat completions:

```bash
curl -s http://localhost:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "meow-lite", "messages": [{"role": "user", "content": "Explain gravity"}]}'
```

Anthropic-compatible messages:

```bash
curl -s http://localhost:8011/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model": "meow-lite", "max_tokens": 100, "messages": [{"role": "user", "content": "Explain gravity"}]}'
```

Both support `"stream": true` for server-sent events (OpenAI: `data: [DONE]` terminator;
Anthropic: message_start/content_block_delta/message_stop events).

Responses are fully deterministic: the same prompt always produces the same meows,
seeded from a sha256 of the prompt. Cats are consistent. Science.

## Endpoints

| Method | Path                    | Purpose                                  |
|--------|-------------------------|------------------------------------------|
| GET    | `/health`               | Liveness check                            |
| GET    | `/v1/models`            | Lists the one true model: `meow-lite`     |
| POST   | `/v1/chat/completions`  | OpenAI-compatible completions (stream ok) |
| POST   | `/v1/messages`          | Anthropic-compatible messages (stream ok) |

## Roadmap

v2 will load a real trained meow model.
