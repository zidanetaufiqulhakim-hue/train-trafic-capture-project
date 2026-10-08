# Deploying to Hugging Face Spaces

The web app runs as a Gradio Space. The repository is laid out so it can be
pushed to a Space as-is — there is no build step and no separate deployment branch.

Live Space: <https://huggingface.co/spaces/zhakim/train-traffic-capture>

## What the Space reads

| File | Purpose |
|---|---|
| `README.md` front matter | Space configuration: `sdk: gradio`, `sdk_version`, `python_version`, `app_file: app.py` |
| `app.py` | Entry point. Puts the project root on the import path and launches the UI from `application/app.py` |
| `requirements.txt` | Python packages installed at build time |
| `packages.txt` | Debian packages installed at build time (`ffmpeg`, used by yt-dlp to merge YouTube streams) |
| `model_registry/passenger_capture_yolov8s.pt` | Model weights, stored with Git LFS |

`api/`, `docs/` and `notebooks/` are ignored by the Space runtime. Training
data and run artifacts are excluded by `.gitignore`.

## Prerequisites

- A Hugging Face account and an access token with **write** permission
  (Settings → Access Tokens).
- Git LFS: `git lfs install` (once per machine). `.gitattributes` already routes
  `*.pt` through LFS; Spaces reject files over 10 MB that are not in LFS.

## First deployment

1. Create the Space at <https://huggingface.co/new-space>: SDK **Gradio**,
   hardware **CPU basic** (free) or **ZeroGPU**.

2. Add the Space as a git remote and push:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```

   Git asks for a username (your Hugging Face username) and a password (the
   access token).

   A newly created Space already contains an initial commit, so the first push
   is rejected as non-fast-forward. Since this repository is the source of
   truth, overwrite it once with `git push --force space main`. **This replaces
   everything in the Space**, so only do it on a Space whose contents you do not need.

3. Open the Space's **Logs** tab. The build installs the requirements (several
   minutes the first time — PyTorch is large), then the app starts. The status
   line in the UI reads "Models ready" once YOLO and EasyOCR are loaded.

## Updating

```bash
git push space main
```

Every push rebuilds and restarts the Space. Keep `origin` (GitHub) and `space`
in step by pushing to both.

## Configuration

Set these under the Space's **Settings → Variables and secrets**. None are required.

| Name | Kind | Effect |
|---|---|---|
| `YOUTUBE_COOKIES` | Secret | Contents of a Netscape-format `cookies.txt`. Lets yt-dlp past YouTube's bot check (see below). Never commit cookies to the repository. |
| `DISABLE_PNG_EXPORT` | Variable | `1` hides the "Download chart (PNG)" button. Already hidden automatically on Spaces, where the Chrome binary Kaleido needs is not installed. |
| `YOLO_MODEL_PATH` | Variable | Path to alternative weights inside the repository. |

## Hardware

- **CPU basic** works. Expect roughly one second per sampled frame or slower, so
  a 5-minute recording at 1 fps takes several minutes.
- **ZeroGPU** speeds up detection. The pipeline call is wrapped with
  `spaces.GPU`, and the GPU time requested is sized to the video (30–120 s)
  because ZeroGPU deducts the requested time from each visitor's daily quota
  and rejects long requests from free accounts. EasyOCR stays on CPU. On
  non-ZeroGPU hardware the wrapper has no effect.

## Limitations on Spaces

- **YouTube downloads are often blocked.** YouTube challenges requests from
  datacenter addresses with a sign-in prompt. The app reports this and asks the
  user to upload the file instead. Supplying `YOUTUBE_COOKIES` works around it.
- **Free Spaces sleep** after a period without visitors; the next visit waits
  for a cold start while the models reload.
- **Storage is ephemeral.** Uploads and generated CSVs live in temporary files
  and disappear on restart.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Push rejected: file larger than 10 MB | The weights were committed without LFS. Run `git lfs install`, then `git lfs migrate import --include="*.pt"` and push again. |
| "Model file not found at …" in the status line | The weights are missing from the Space, or exist only as an LFS pointer. Check that `model_registry/passenger_capture_yolov8s.pt` shows an "LFS" badge and its real size (≈22 MB) in the Files tab. |
| Build fails resolving packages | `sdk_version` in `README.md` and the `gradio` pin in `requirements.txt` must match. |
| Configuration error banner | Invalid front matter. `colorFrom` / `colorTo` accept only red, yellow, green, blue, indigo, purple, pink or gray. |
| Analysis stops partway on ZeroGPU | The video needed more than the 120 s GPU allowance. Lower the sampling rate or use a shorter clip. |
