"""
Video input: open a local file or download from YouTube, then sample frames.

The only public contract is:
  open_video(path)          → cv2.VideoCapture  (raises ValueError if unreadable)
  iter_sampled_frames(cap)  → yields (timestamp_sec, frame)
  count_sampled_frames(path) → int  (how many frames will be processed)
  download_youtube_video(url) → (video_path, temp_dir)  (caller must delete temp_dir)
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# ── Public API ────────────────────────────────────────────────────────────────


def open_video(path: str) -> cv2.VideoCapture:
    """Opens a video file. Raises ValueError with a clear message if it cannot be read."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video at: {path}")
    return cap


def iter_sampled_frames(
    cap: cv2.VideoCapture, target_fps: float
) -> Iterator[tuple[float, np.ndarray]]:
    """Yields (timestamp_seconds, frame) for every Nth frame.

    N is chosen so the effective sampling rate matches target_fps.
    The caller is responsible for releasing the VideoCapture when done."""
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_interval = max(1, round(video_fps / target_fps))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            yield frame_idx / video_fps, frame
        frame_idx += 1


def count_sampled_frames(video_path: str, target_fps: float) -> int:
    """Returns the number of frames that will be sampled, for progress tracking."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 1  # avoid division by zero downstream
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    interval = max(1, round(video_fps / target_fps))
    return max(1, total // interval)


def is_youtube_url(text: str) -> bool:
    return "youtube.com" in text or "youtu.be" in text


def download_youtube_video(url: str) -> tuple[str, str]:
    """Downloads a YouTube video to a temporary directory.

    Returns (video_file_path, temp_dir).
    The caller must delete temp_dir after the video has been processed.

    Raises RuntimeError if yt-dlp is not installed or the download fails."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is required for YouTube downloads. "
            "Install it with: pip install yt-dlp"
        )

    tmp_dir = tempfile.mkdtemp(prefix="train_traffic_yt_")
    output_template = os.path.join(tmp_dir, "video.%(ext)s")

    # Locate ffmpeg inside the active Python environment so yt-dlp can merge
    # video+audio streams even when ffmpeg is not on the system PATH.
    ffmpeg_location = _find_ffmpeg()

    ydl_opts = {
        # Prefer a pre-merged mp4 first (no ffmpeg needed).
        # Fall back to separate streams merged by ffmpeg if a pre-merged file
        # is unavailable or lower quality than acceptable.
        "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    # YouTube blocks datacenter IPs (e.g. HF Spaces) with "Sign in to confirm
    # you're not a bot". A Netscape-format cookies.txt supplied through the
    # YOUTUBE_COOKIES secret gets past that. Never commit the cookies themselves.
    cookies = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if cookies:
        cookie_path = os.path.join(tmp_dir, "cookies.txt")
        with open(cookie_path, "w") as f:
            f.write(cookies + "\n")
        ydl_opts["cookiefile"] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
    except Exception as e:
        if "not a bot" in str(e):
            raise RuntimeError(
                "YouTube blocked the download from this server (bot check). "
                "Download the video yourself and use the Local file tab instead."
            )
        raise RuntimeError(f"YouTube download failed: {e}")

    # yt-dlp may produce a file with a slightly different name after merging
    if not os.path.exists(filename):
        candidates = list(Path(tmp_dir).glob("video.*"))
        if not candidates:
            raise RuntimeError(f"YouTube download produced no file. Check the URL: {url}")
        filename = str(candidates[0])

    return filename, tmp_dir


def _find_ffmpeg() -> str | None:
    """Returns the path to the ffmpeg binary, checking the active Python
    environment's bin directory before falling back to the system PATH.

    yt-dlp needs this when ffmpeg is installed inside a conda/venv environment
    but not on the global system PATH."""
    import shutil

    # Check alongside the running Python interpreter first (covers conda envs)
    python_bin = Path(sys.executable).parent
    for candidate in [python_bin / "ffmpeg", python_bin / "ffmpeg.exe"]:
        if candidate.exists():
            return str(candidate)

    # Fall back to whatever is on PATH
    return shutil.which("ffmpeg")
