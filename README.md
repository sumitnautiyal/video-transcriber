# video-transcriber

I was cleaning up my laptop before reinstalling Windows and found about 17 GB of recorded seminars, classes and info sessions that I would never rewatch but didn't want to lose. Instead of backing up the videos, I turned them into searchable text with this script. It uses [WhisperX](https://github.com/m-bain/whisperX) for transcription and word alignment, and optionally [pyannote](https://github.com/pyannote/pyannote-audio) to label who is speaking. Output is one markdown file per video.

It's a hobby project, tested only on Windows 11 with an NVIDIA RTX 3060 (6 GB) and Python 3.13.

## Requirements

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on your PATH
- For speaker labels: a Hugging Face token, with the terms accepted for
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and
  [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- An NVIDIA GPU is optional. Without one it falls back to CPU, which is slower.

## Setup

```
uv sync
cp .env.example .env    # then put your token in .env
```

`pyproject.toml` pulls torch from the CUDA 12.4 wheel index. On a machine without CUDA, edit or remove the `[tool.uv.sources]` section.

## Usage

```
uv run python transcribe.py path/to/videos
uv run python transcribe.py talk.mkv --model small --out transcripts
uv run python transcribe.py path/to/videos --language en    # skip language auto-detection
uv run python transcribe.py path/to/videos --no-diarize     # no token needed
```

Folders are searched recursively. A file that already has a `*_transcript.md` is skipped, so you can stop and restart a long run. Videos with no detected speech get a transcript saying so. Failures are listed at the end and the run continues past them.

Example output, from a short text-to-speech clip I generated (see [examples/sample.md](examples/sample.md)):

```
# Transcript: sample.wav

- Language: en
- Mean word confidence: 86.1%
- Processing time: 6 s

---

### SPEAKER_00
Welcome everyone. Today we are testing a small transcription script. It converts recorded talks into searchable text. Thank you for listening.
```

## What I measured

I ran the first version (openai-whisper, CPU, no alignment or speakers) and then this one (WhisperX on the GPU, with alignment and diarization) over the same recordings, both with the `base` model. Timings below come from the transcript files' modification times and the processing time WhisperX writes into each transcript, so they are wall-clock times from real runs, not a controlled benchmark.

| | |
|---|---|
| Recordings transcribed with WhisperX | 25 (31.9 hours of video, 17.0 GiB) |
| Total transcript size | 1.4 MB |
| Sum of WhisperX processing times | 6.4 hours (median 6.7 min per file, range 19 s to 70 min) |

CPU version vs GPU version on the 23 recordings both completed in an uninterrupted run (30.4 hours of video):

| | CPU, plain Whisper | GPU, WhisperX + speakers |
|---|---|---|
| Total time | 3.2 h | 6.3 h |
| Speed vs. real time | 9.5x | 4.9x |

The GPU total is dominated by one stretch. On 16 of the recordings the GPU version took 1.7 h against 2.3 h for the CPU version (13x vs 9.7x real time, while also doing alignment and diarization). On the other 7 it took 4.6 h against 0.9 h, about 1.8x real time. I did not investigate that stretch. The laptop's GPU fan is failing and the GPU was hitting 100%, so thermal throttling is a plausible cause, but I haven't verified it. So in practice the GPU version was not clearly faster in my run; the main gain is speaker labels and word timestamps.

- The transcripts are over 99.9% smaller than the videos, but they contain only speech: slides, screen shares and anything visual are gone.
- "Mean word confidence" is the average alignment score WhisperX gives each word. It is a rough signal, not an accuracy measure. Clean recordings scored around 75-85%.

## Known problems

- **Language auto-detection is unreliable.** One English-language recording was labelled Norwegian (`nn`) and came out with 41% mean confidence. Another, a casual Hindi/English conversation, was labelled Hindi with 48%. Use `--language en` if you know the language.
- The `base` model is fast but makes mistakes on names and technical terms. Try `--model small` or larger if accuracy matters more than time.
- Speaker labels are `SPEAKER_00`, `SPEAKER_01`, ... and are not always right, especially with overlapping speech or laughter. One casual conversation produced pages of repeated "Ha ha ha".
- `apply_compat_shims()` in `transcribe.py` patches around incompatibilities between whisperx/pyannote and recent torch, torchaudio and huggingface_hub releases. One of them turns off torch's `weights_only` safety check for `torch.load`, so only use it with models you trust. It may be unnecessary with newer whisperx versions.
- No tests.

## License

MIT
