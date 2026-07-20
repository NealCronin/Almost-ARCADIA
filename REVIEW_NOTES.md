# Review notes and discussion points

This build intentionally simplifies the model configuration boundary. The changes below are worth keeping in mind during real-hardware testing.

## 1. Raw argument parsing

The argument field treats unquoted spaces, commas, and line breaks as separators. A literal comma or a value containing spaces must be quoted. For example:

```text
--tensor-split "1,1"
--chat-template-kwargs '{"preserve_thinking": false}'
```

This is deterministic and shell-free, but it is not identical to pasting a complete shell command. The application passes parsed tokens directly to `Popen` with `shell=False`.

## 2. Native argument responsibility

Only app-owned routing and source flags are rejected. Almost ARCADIA does not maintain a duplicate schema for GPU placement, batching, KV cache types, speculative decoding, or future llama.cpp flags. An invalid native value is therefore discovered when `llama-server` starts, and the actionable error is in the service log.

This is the main tradeoff that removes the previous form/validation overhead.

## 3. Hugging Face ambiguity

Repository-only input is convenient for a repository with one usable model. Repositories containing several quants require an exact `blob` or `resolve` link. Exact links also preserve nested repository paths and revisions. This avoids guessing which quant the user intended.

## 4. Draft models

Draft-model controls were removed from the typed interface. Native llama.cpp flags such as `--spec-draft-hf`, `--spec-draft-model`, and `--spec-type` can be placed in Additional Arguments. Consequently:

- llama.cpp owns draft-model download/resolution when an HF draft flag is used.
- Almost ARCADIA does not display draft download progress or validate a draft repository in advance.
- A local `--spec-draft-model` path must exist on the compute host, not merely on the Django client.

A future dedicated draft source field would be justified only if the app needs unified cache/progress management.

## 5. Instruction IP versus inference IP

The compute node stores an instruction-server address. Each model also stores an editable inference IP that defaults to the node address. This supports hosts with different control and data interfaces, but introduces a valid failure mode: the client cannot prove that a remote inference IP belongs to the remote machine. Failed binds are reported by that host's `llama-server` log.

## 6. Graph Agent compatibility

The previous branch implementation did not match the pinned Priority Map contract. This build uses the exact constructor/lifecycle shape from commit `ea6d1064175b20c1e90dd3f1ffb0b4173f68e03d`:

- positional `graph_builder`, `task_description`
- no-argument `should_run`, `start_async_if_ready`, `poll_finished`, and `update_priorities`
- single-worker asynchronous inference
- `apply_score_delta` and `mark_agent_reviewed` graph mutations
- graph-agent shutdown before graph-builder shutdown

This is unit-testable, but a real end-to-end graph-agent run still needs a real Logical LLM, Priority Map installation, and SAM3 checkpoint.

## 7. Service replacement

Services are owned by the controller that launched them. Replacing a configured port stops the owned child first. If the replacement fails, the previous process is not reconstructed automatically. This avoids killing unrelated processes or pretending rollback succeeded, but a failed replacement leaves the port stopped until a subsequent start.

## 8. Security boundary

The instruction API still accepts no arbitrary executable or shell command. Raw argument flexibility does not make the command a shell command. However, the application remains intended only for trusted users: inference and instruction endpoints have no authentication or TLS.

## 9. Heavyweight validation still required

The supplied unit checks do not claim:

- a successful build on every Metal/CUDA combination;
- a real multimodal model/projector request;
- a real SAM3 checkpoint prediction;
- a complete Priority Map moving sequence with graph-agent updates;
- remote firewall/VPN compatibility.

Recommended first real tests:

1. Logical text model from one exact GGUF link.
2. Logical vision model plus exact projector link.
3. Separate Visual LLM on a different port.
4. A remote compute host with instruction and inference health checks.
5. SAM3 with a real checkpoint.
6. Priority Map with Graph Agent disabled.
7. Priority Map with Graph Agent enabled and enough graph growth to trigger it.


## 10. Model alias ownership

Almost ARCADIA no longer emits `--alias` and no longer stores role aliases such as `logical-model` or `visual-model`. The request client discovers the model ID from the running server's `/v1/models` endpoint. A user who wants a custom alias can provide `--alias <name>` in Additional Arguments, and that exact server-provided ID is used for inference. Older saved `model_alias` keys are ignored during validation for migration compatibility.

## 11. Remote reachability versus model readiness

A green compute-node test proves only that the instruction server answers on its control port. It does not prove that `llama-server` exists on the remote host, the GGUF can be downloaded, the requested inference IP can be bound, the inference port is permitted through the firewall, or the model can fit in memory. Remote service startup now has a long control-plane timeout, reuses an identical running process, maps startup failures to a detailed HTTP 502 response, and includes the startup log tail where available.

## 12. Hugging Face cache layout

Each compute node now owns this app-local cache layout by default:

```text
workspace/huggingface/
├── models/
└── mmproj/
```

`ARCADIA_MODELS_DIR` overrides the `workspace/huggingface` root, not either child directory. Existing files under the older `workspace/models/...` layout are not moved automatically; move them manually or allow Hugging Face to populate the new cache.

## 13. SAM3 test and mask transport

The SAM3 card now has an image-and-concept test that returns a rendered PNG. It deliberately calls the same `/v1/predict` path and `SAMClient.segment()` method used by Priority Map, so it exercises the configured compute node, checkpoint, confidence, and raw-mask response rather than a separate preview-only implementation.

The current raw mask protocol serializes full masks as JSON arrays. This is straightforward and keeps the pipeline implementation simple, but it can be expensive for large images and many masks. A later transport revision should consider lossless mask PNGs, compact run-length encoding, or a binary response envelope while retaining labels, scores, and boxes.

## SAM3 checkpoint browser

A browser cannot expose an arbitrary absolute path from either the client computer or a remote compute node. The Browse control therefore uploads the selected `.pt` checkpoint to the selected compute node rather than pretending the browser path is usable. Local and remote writes are streamed and atomic, and the resulting absolute compute-node path is placed into the checkpoint field. The default upload destination is `workspace/huggingface/models`; `ARCADIA_MODELS_DIR` changes the Hugging Face root.
