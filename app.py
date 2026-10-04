"""Entry point for Hugging Face Spaces deployment.

HF Spaces auto-detects this file at the repo root and executes it on boot.
Locally, `python app.py` and `python application/app.py` both still work.

ZeroGPU
-------
On HF Spaces with ZeroGPU hardware, the pipeline generator runs inside a
@spaces.GPU-decorated wrapper. A GPU is dynamically allocated for the
duration of a single video analysis (up to 300 s) and released afterwards.

The import ordering below is deliberate:
  1. Import the runner MODULE so we can rebind an attribute on it.
  2. If `spaces` is installed, replace runner.iter_pipeline with the
     GPU-wrapped generator.
  3. Then import application.app — its `from runner import iter_pipeline`
     picks up the already-patched function.

On local machines without the `spaces` package, the ImportError branch
is taken and the original (unwrapped) generator is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.pipeline import runner as _runner

try:
    import spaces

    _original_iter_pipeline = _runner.iter_pipeline

    @spaces.GPU(duration=300)
    def _iter_pipeline_gpu(video_path, models, target_fps):
        """ZeroGPU-decorated generator — holds a GPU for the pipeline run."""
        yield from _original_iter_pipeline(video_path, models, target_fps)

    _runner.iter_pipeline = _iter_pipeline_gpu
except ImportError:
    pass

from application.app import app, _CSS, _THEME  # noqa: E402

if __name__ == "__main__":
    app.launch(show_error=True, theme=_THEME, css=_CSS)
