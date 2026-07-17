# Audit Report: Almost-ARCADIA-next-draft vs neal-branch

**Date:** 2026-07-17
**Source:** `/Users/neal/Downloads/Almost-ARCADIA-next-draft.zip`
**Baseline:** `/Users/neal/Documents/Almost-ARCADIA` @ `3e6de2d71f2ecece7f80113c7d27fe488072f4c3`
**Draft claims base:** `28d4dfd966ee0066471e418bdddafb0f9960c98b`
**AUDIT_ROOT:** `/var/folders/gt/dyd1npbs7yzd5ysdcgmpbrdm0000gn/T/almost-arcadia-next-draft.U07Da9`
**Review worktree:** `AUDIT_ROOT/review` (branch `review/next-draft-audit-fixes`)

---

## 1. Complete Diff and Changed Files

- **Complete diff:** `AUDIT_ROOT/complete.diff` (10,965 lines, 85 changed-file entries)
- **Changed-files manifest:** `AUDIT_ROOT/changed-files.txt` (85 entries)
- Files added: 20 (including `llm_roles.py`, `model_sources.py`, `orchestration.py`, installers, scripts, templates)
- Files removed: 27 (including `networking.py`, `llm_settings.py`, `web/artifacts.py`, `web/tools.py`, `web/uploads.py`, `web/models.py`, `web/admin.py`, 16 test files, `migrations/`, templates)
- Files modified: 38 (including `core/config.py`, `llm_runtime.py`, `views.py`, `forms.py`, `default_config.json`, `pyproject.toml`)

## 2. Critical Architecture Changes

### 2.1 Native llama.cpp Cutover (LLM Runtime)
- **From:** `llama-cpp-python` Python server (`sys.executable -m llama_cpp.server`)
- **To:** Native `llama-server` binary (C++ executable)
- `find_executable()` replaces `models_directory()` — probes `ARCADIA_LLAMA_SERVER` env var, `vendor/llama.cpp/build/bin/llama-server`, and PATH
- Flag names changed from llama-cpp-python underscore style (`--n_ctx`) to native style (`--ctx-size`, `-ngl`)
- `build_command()` now takes `models_dir` parameter, delegates model resolution to `HuggingFaceModelResolver`
- **Verified:** Every emitted flag is supported by local `/opt/homebrew/bin/llama-server` (build 9810)

### 2.2 Logical/Visual LLM Role Separation
- New `VisualLLMMode` type: `"same_as_logical" | "separate"`
- `LLMRoleResolver.resolve()` maps role to `ConfiguredService` supporting shared mode
- `LLMRoleResolver.validate_pair()` checks port collision for separate mode only
- Legacy `llm` service key auto-migrated to `logical_llm` in `PriorityMapToolConfig.from_dict()`
- `visual_llm_mode` default: `"same_as_logical"` if `logical_llm` has `vision_enabled`, else `"separate"`

### 2.3 Priority Map Adapter Simplification
- **From:** 496 lines with internal `RemoteSceneUnderstanding`, `RemoteSegment`, optical flow, SAM integration, video extraction
- **To:** 61 lines — thin pass-through to external `priority_map.runner.PriorityMapRunner`
- Signature changed from `llm_client` to `logical_llm_client` + `visual_llm_client`
- `PipelineResult` returns `(frames_processed, output_paths)`

### 2.4 Configuration Restructuring
- Base: `{services: {llm, sam3}, pipeline, output_root}` — flat
- Draft: `{tools: {"priority-map": {services, visual_llm_mode, pipeline, output}}}` — nested
- `default_config.json` now has empty `services: {}` — no pre-configured LLM

### 2.5 Web View/Form Restructuring
- URL patterns reduced from 23 to 20 — removed uploads, artifacts, runs, results, endpoint test, service lifecycle
- Analysis page stripped to bare status (no pipeline config, uploads, or results)
- New: role-parameterized LLM save, test chat with image upload, repository inspection
- Service lifecycle moved from explicit start/stop routes to orchestrator-based `ensure_llm_role`/`ensure_configured_service`

## 3. Blocking Defects

### 3.1 ConfigStore.load() — Lost Error Wrapper (Regression — FIXED IN REVIEW)
**File:** `core/config.py` — `ConfigStore.load()`
**Status:** Regression in draft; **fixed** in review worktree branch `review/next-draft-audit-fixes`.
**Impact:** A corrupt or unreadable `default_config.json` propagates a raw `OSError`/`JSONDecodeError` instead of a wrapped `ConfigurationError`.
**Draft had:**
```python
payload = self.default_path.read_text(encoding="utf-8")
config = AppConfig.from_dict(json.loads(payload))
```
**Fix applied (review worktree):**
```python
try:
    payload = self.default_path.read_text(encoding="utf-8")
    config = AppConfig.from_dict(json.loads(payload))
except (OSError, json.JSONDecodeError) as exc:
    raise ConfigurationError(f"Could not read default configuration {self.default_path}: {exc}") from exc
```

### 3.2 Combined Test Suite Collection Failure (Regression)
**Status:** Regression
**Evidence:** Running `pytest REPO_ROOT/tests DRAFT_ROOT/tests` with draft imports produces 4 collection errors. The old `test_config.py` and `test_runtime.py` at `REPO_ROOT/tests/` import from deleted modules (`core.services.llm_roles`, `core.services.model_sources`). Both old and draft test files share the same filenames, causing import conflicts.
**Affected:** Any test suite that includes both old and draft test directories.

### 3.3 Removed Modules Break Existing Callers (Regression)
**Status:** Regression
**Removed modules:** `core/networking.py`, `core/services/llm_settings.py`, `web/artifacts.py`, `web/tools.py`, `web/uploads.py`, `web/models.py`, `web/admin.py`
**Affected:** Any external code importing from these modules. The old test files reference these modules. Internal code has been refactored to use new equivalents.

### 3.4 Removed Workflow Endpoints (Removed Feature)
**Status:** Intentional removal (documented)
**Removed:** Upload management, results page, streaming, endpoint test, artifact serving, run management, pipeline configuration forms
**Impact:** Users cannot upload files, view results, configure pipeline settings, or test arbitrary endpoints through the UI. The draft description says "review before production use" for these.

### 3.5 Priority Map Runner Dual-Client Contract (Incomplete)
**Status:** Incomplete — cannot verify
**Evidence:** `priority_map` is not installed in the audit environment. The draft's `PriorityMapAdapter` expects `logical_llm_client` and `visual_llm_client` parameters, but the installed `PriorityMapRunner` constructor signature cannot be verified. The draft's adapter is a thin pass-through, so if the installed runner doesn't support dual clients, the adapter will fail at runtime.

### 3.6 Remote Uncached-Model Cache Reuse (Unverified)
**Status:** Unverified — no test written
**Evidence:** The `HuggingFaceModelResolver` downloads model/projector/draft files into `models_dir` subdirectories. The draft's `build_command` always calls `resolver.resolve(settings)`, which downloads on cache miss. No test exercises a cache-miss-then-cache-hit sequence on a remote node. The resolved paths are always local regardless of node mode, so remote hosts receive local paths — this is correct per the draft's architecture (the instruction server, not the controller, owns remote paths), but the cache-reuse assertion remains untested.

## 4. Unsafe Behavior

### 4.1 `install_macos_metal.command` — No Error Handling
**File:** `install_macos_metal.command`
**Issue:** The macOS Finder wrapper does not set `-e`, so its final `read -r _` prompt can mask a failed child command. The underlying `install_macos_metal.sh` does use `set -euo pipefail`.

### 4.2 `install_windows_cuda.ps1` — No Explicit Stop-on-Error Beyond `$ErrorActionPreference`
**Status:** `$ErrorActionPreference = "Stop"` is set at the top, so errors are caught. However, `run_windows.ps1` lacks any stop-on-error setting.

### 4.3 Remote Node Trust Boundary
**Status:** Preserved (acceptable for draft)
**Evidence:** The draft's `ensure_configured_service` delegates to `InstructionClient` for remote nodes. Model resolution is always host-local (`HuggingFaceModelResolver` downloads to local `models_dir`). No remote-path injection exists. The remote instruction server, not request configuration, is the authority for what runs.

## 5. Native Flag Verification

### 5.1 llama-server Binary
- **Path:** `/opt/homebrew/bin/llama-server`
- **Version:** 9810 (2f18fe13c)
- **Build:** Darwin arm64

### 5.2 Flag Audit
| Draft Flag | Native Support | Notes |
|---|---|---|
| `-m` / `--model` | ✓ | `--model FNAME` |
| `--host` | ✓ | Server param |
| `--port` | ✓ | Server param |
| `--ctx-size` | ✓ | `-c, --ctx-size N` |
| `-ngl` | ✓ | `--gpu-layers, --n-gpu-layers N` |
| `--threads` | ✓ | `-t, --threads N` |
| `--batch-size` | ✓ | `-b, --batch-size N` |
| `--ubatch-size` | ✓ | `-ub, --ubatch-size N` |
| `--cache-type-k` | ✓ | `-ctk, --cache-type-k TYPE` |
| `--cache-type-v` | ✓ | `-ctv, --cache-type-v TYPE` |
| `--cache-type-k-draft` | ✓ | `--spec-draft-type-k, -ctkd, --cache-type-k-draft TYPE` |
| `--cache-type-v-draft` | ✓ | `--spec-draft-type-v, -ctvd, --cache-type-v-draft TYPE` |
| `--alias` | ✓ | `-a, --alias STRING` |
| `--chat-template` | ✓ | `--chat-template JINJA_TEMPLATE` |
| `--mmproj` | ✓ | `-mm, --mmproj FILE` |
| `-md` / `--model-draft` | ✓ | `--spec-draft-model, -md, --model-draft FNAME` |
| `--flash-attn` | ✓ | `-fa, --flash-attn [on\|off\|auto]` |
| `--mmap` / `--no-mmap` | ✓ | `--mmap, --no-mmap` |
| `--mlock` | ✓ | `--mlock` |

**All 17 draft flags are supported by the installed binary.** No regressions found in flag emission.

## 6. Installer Syntax Verification

### 6.1 `install_macos_metal.sh`
- **Syntax:** `bash -n` — PASS (no errors)
- **Behavior:** `set -euo pipefail`, checks Darwin/arm64, Xcode CLI, Homebrew, installs cmake/ninja/git/python3.12, creates venv with `.[all]`, shallow-clones llama.cpp, builds with `GGML_METAL=ON`, `GGML_NATIVE=ON`, runs `verify_llama_flags` and `manage.py check`
- **Idempotency:** `.venv` check + `git fetch --depth 1 origin master && reset --hard origin/master` pattern

### 6.2 `install_macos_metal.command`
- **Syntax:** `bash -n` — PASS (no errors)
- **Issue:** No `-e` flag; exit status of child script is masked by final `read -r _`

### 6.3 `install_windows_cuda.ps1`
- **Syntax:** Parsed (no errors)
- **Behavior:** `$ErrorActionPreference = "Stop"`, winget-based dependency bootstrap, venv with `.[all]`, CUDA build with `GGML_CUDA=ON`, `LLAMA_BUILD_EXAMPLES=ON`

## 7. Test Results

### 7.1 Draft's Own Checks (from DRAFT_ROOT)
| Check | Result |
|---|---|
| `pytest -q` (15 draft tests) | ✅ **15 passed** |
| `ruff format --check .` | ✅ **41 files already formatted** |
| `ruff check .` | ✅ **All checks passed** |
| `mypy core web project manage.py` | ✅ **No issues in 34 source files** |
| `python manage.py check` | ✅ **System check identified no issues** |

### 7.2 Focused Audit Tests (from review worktree)
| Check | Result |
|---|---|
| `test_shared_role_resolution` | ✅ Shared role resolves both logical and visual to same service |
| `test_separate_role_collision_rejected` | ✅ Same node + same port raises ConfigurationError |
| `test_separate_role_different_port_accepted` | ✅ Different ports on same node accepted |
| `test_separate_role_different_node_accepted` | ✅ Same port on different nodes accepted |
| `test_legacy_llm_migration` | ✅ `llm` -> `logical_llm` with port preserved |
| `test_native_flag_emission` | ✅ All expected flags emitted correctly |
| `test_build_command_with_vision_and_draft` | ✅ `--mmproj` and `-md` with correct paths |
| `test_validate_pair_called_at_save_boundary` | ✅ `validate_pair` called in `save_llm` and `save_visual_mode` |

### 7.3 Full Suite (Review Worktree)
| Check | Result |
|---|---|
| `pytest tests/` (23 tests: 15 draft + 8 focused) | ✅ **23 passed** |
| `ruff format --check .` | ✅ **42 files already formatted** |
| `ruff check .` | ✅ **All checks passed** |
| `mypy core web project manage.py` | ✅ **No issues in 34 source files** |
| `python manage.py check` | ✅ **System check identified no issues** |

### 7.4 Combined Test Suite (Draft + Old Tests)
| Check | Result |
|---|---|
| `pytest REPO_ROOT/tests DRAFT_ROOT/tests` | ❌ **4 collection errors** — `test_config.py` and `test_runtime.py` from both trees conflict; old tests import from deleted modules |

## 8. Recommended Patches

### 8.1 Critical (Must Fix Before Merge)

1. **Restore ConfigStore error wrapper** — `core/config.py` `ConfigStore.load()`: add try/except around `self.default_path.read_text` + `json.loads` wrapping raw errors in `ConfigurationError`. **Already applied** in review worktree branch `review/next-draft-audit-fixes`.

2. **Remove or reconcile old test files** — The old `tests/test_config.py` and `tests/test_runtime.py` at `REPO_ROOT/tests/` are incompatible with the draft's module structure. Either: (a) replace them with the draft versions, or (b) remove them if the draft's own tests provide equivalent coverage.

3. **Remove old test files and modules** — 16 deleted test files, `web/artifacts.py`, `web/tools.py`, `web/uploads.py`, `web/models.py`, `web/admin.py`, `core/networking.py`, `core/services/llm_settings.py` must be removed from the target branch before merge.

### 8.2 High Priority

4. **Add `-e` to `install_macos_metal.command`** wrapper to propagate failures.

5. **Add `$ErrorActionPreference = "Stop"` to `run_windows.ps1`** for consistency.

6. **Verify Priority Map runner dual-client contract** — The draft's `PriorityMapAdapter` requires `logical_llm_client` and `visual_llm_client`. Before merging, install `priority-map` and verify the `PriorityMapRunner` constructor accepts both parameters.

7. **Add remote cache-reuse test** — Write a test that calls `HuggingFaceModelResolver.resolve` with a mocked `hf_hub_download` that records calls, then calls it twice with the same repo_id asserting the second call is served from cache (no download). This was not implemented in this audit.

### 8.3 Medium Priority

8. **Verify `validate_pair` is called at pre-launch boundaries** — Currently only called at save (in `save_llm` and `save_visual_mode` views). Add a pre-launch check in `ensure_llm_role` or `ensure_configured_service`.

## 9. Summary

| Category | Count |
|---|---|
| Test-passing defects | 0 |
| Blocking regressions | 4 (ConfigStore wrapper — fixed in review, combined test suite, removed modules, removed workflows) |
| Incomplete/unverified | 2 (Priority Map dual-client contract, remote cache reuse) |
| Unsafe behavior | 2 (installer command wrapper, PowerShell launcher) |
| Native flag regressions | 0 (all 17 flags verified) |
| Installer syntax errors | 0 (all pass) |
| Draft's own checks pass | 5/5 ✅ |
| Focused audit tests pass | 8/8 ✅ |
| Full suite (review worktree) | 23/23 ✅ |