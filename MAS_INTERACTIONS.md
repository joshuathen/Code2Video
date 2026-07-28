# MAS Gemini Interactions API

The new multi-agent system (MAS), cross-run belief reflection, and MAS
evaluation use the Google Gemini Interactions API.

The original Code2Video agent remains on its existing API implementation in
`src/gpt_request.py`.

## MAS request boundary

`src/mas_interactions.py` owns the Interactions API integration:

- text and structured-output interactions;
- explicit function calling via `function_call` and `function_result` steps;
- continuation with `previous_interaction_id`;
- video and image input through the Gemini Files API or inline data;
- interaction token-usage normalisation; and
- compatibility response fields used by the existing MAS parsers and logs.

The following MAS operations use this adapter:

- outline generation;
- Script Writer, Animation Planner, Coder, and Orchestrator invocations;
- ScopeRefine repair calls made by the MAS;
- rendered-video review;
- storyboard asset enhancement calls;
- post-run belief generation and updating; and
- AES/TQ evaluation when the `mas` runner is selected.

## Configuration

Set `GEMINI_API_KEY` (preferred) or `API_KEY`. The existing
`gemini.api_key` value in `src/api_config.json` remains a fallback.

The Interactions API requires `google-genai>=2.3.0`.

### Setonix installation

The MAS launcher executes Python inside the existing Manim container. Install
the Interactions dependencies into that same Python 3.11 environment:

```bash
module load singularity/4.1.0-nompi

singularity exec \
  /software/projects/pawsey1357/jthen/containers/manim-0.19.0.sif \
  /software/projects/pawsey1357/jthen/venvs/code2video-manim/bin/python \
  -m pip install -U -r src/requirements-mas-interactions.txt
```

Do not run `pip install -U -r src/requirements.txt` on the Setonix login-node
Python solely for this migration. That file describes the complete Code2Video
environment and causes pip to build ManimPango, which requires the native
`pangocairo` development files. The container already supplies the Manim
runtime required by the MAS jobs.

Verify the environment used by the jobs:

```bash
singularity exec \
  /software/projects/pawsey1357/jthen/containers/manim-0.19.0.sif \
  /software/projects/pawsey1357/jthen/venvs/code2video-manim/bin/python \
  -c 'from importlib.metadata import version; from google import genai; print(version("google-genai")); print(hasattr(genai.Client(api_key="test"), "interactions"))'
```

Model environment variables remain:

- `MAS_MODEL`
- `CODE_MODEL`
- `REPAIR_MODEL`
- `EVAL_MODEL`

Agent tool calls are executed by an explicit application-side loop because the
Interactions API does not provide Python automatic function calling. Each
interaction ID and its combined usage are written to the MAS call log.
