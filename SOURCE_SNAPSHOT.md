# Source snapshot

This exported codebase was prepared from the architecture and files on:

```text
Repository: NealCronin/Almost-ARCADIA
Branch:     neal-branch
Commit:     9a2268a0144bafba92f3249c95b914b7875a6242
Message:    Model Settings UI Fix
```

The ZIP intentionally does not include Git history, runtime caches, virtual environments, downloaded models, generated configuration, logs, uploads, or analysis outputs.

The main replacement in this export is the simplified LLM settings boundary described in `README.md`. It also includes the pinned Priority Map Graph Agent compatibility repair described in `REVIEW_NOTES.md`.

Version `0.3.0` is the next local iteration of that export. It adds the `workspace/huggingface/{models,mmproj}` cache layout, the SAM3 image-and-text segmentation test, and SAM3 semantic-predictor compatibility improvements. These changes have not been pushed to the source branch by this ZIP-generation workflow.

Version `0.4.0` adds streamed local/remote SAM3 checkpoint upload from the Model settings page. The browser-selected `.pt` file is atomically stored in the selected compute node's `workspace/huggingface/models` directory, and its actual compute-node path is written into the saved settings form.

Version `0.5.0` replaces repository `workspace` storage with `huggingface/{models,mmproj}` and OS state directories, adds Host-page SAM3 checkpoint controls, and unifies the SAM3 FastAPI/client contract around compact PNG mask responses.
