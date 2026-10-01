"""Batch-transcribe videos with WhisperX, optionally labelling speakers with pyannote.

Usage:
    python transcribe.py PATH [PATH ...] [--model base] [--language en] [--out DIR] [--no-diarize]

PATH can be a media file or a folder (searched recursively). For each file a
markdown transcript is written next to it (or into --out). Files that already
have a transcript are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

EXTENSIONS = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4a", ".mp3", ".wav", ".flac"}
SUFFIX = "_transcript.md"


def apply_compat_shims():
    """Patch a few libraries so whisperx and pyannote work with recent versions of them.

    These were needed with the versions pip/uv installed for me (torch 2.6,
    pyannote 3.4). Newer whisperx releases may not need them.
    """
    import huggingface_hub
    import torch
    import torchaudio

    # torchaudio >= 2.9 removed these; pyannote still references them.
    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            pass

        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    # torch >= 2.6 defaults torch.load to weights_only=True, which rejects the
    # pyannote checkpoints. This disables that check process-wide, so only run
    # this with models you trust (here: the pyannote models from Hugging Face).
    original_load = torch.load

    def load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = load

    # huggingface_hub >= 0.25 dropped use_auth_token in favour of token.
    original_download = huggingface_hub.hf_hub_download

    def download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return original_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = download
    if hasattr(huggingface_hub, "file_download"):
        huggingface_hub.file_download.hf_hub_download = download


def find_media(paths):
    """Return a list of media files from the given files and folders."""
    files = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.suffix.lower() in EXTENSIONS:
                    files.append(f)
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: {path} not found, skipping", file=sys.stderr)
    return files


def average_confidence(segments):
    """Average of the per-word scores WhisperX gives, or None if there are none."""
    scores = []
    for segment in segments:
        for word in segment.get("words", []):
            if "score" in word:
                scores.append(word["score"])
    if not scores:
        return None
    return sum(scores) / len(scores)


def write_transcript(path, source, language, confidence, seconds, segments):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {source.name}\n\n")
        f.write(f"- Language: {language}\n")
        if confidence is not None:
            f.write(f"- Mean word confidence: {confidence:.1%}\n")
        f.write(f"- Processing time: {seconds:.0f} s\n\n---\n")

        if not segments:
            f.write("\n[No speech detected]\n")
            return

        # Start a new "### SPEAKER_xx" heading whenever the speaker changes.
        current_speaker = None
        for segment in segments:
            speaker = segment.get("speaker")
            if speaker and speaker != current_speaker:
                f.write(f"\n### {speaker}\n")
                current_speaker = speaker
            f.write(segment["text"].strip() + " ")
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Batch-transcribe videos with WhisperX.")
    ap.add_argument("paths", nargs="+", help="media files or folders")
    ap.add_argument("--model", default="base", help="Whisper model size (default: base)")
    ap.add_argument("--out", type=Path, help="write transcripts here instead of next to the source")
    ap.add_argument("--no-diarize", action="store_true", help="skip speaker labels (no HF token needed)")
    ap.add_argument("--language", help="force a language code such as 'en' instead of auto-detecting")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    token = None
    if not args.no_diarize:
        from dotenv import load_dotenv

        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN is not set (see .env.example). Use --no-diarize to skip speaker labels.")

    files = list(find_media(args.paths))
    if not files:
        sys.exit("no media files found")

    apply_compat_shims()
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading '{args.model}' on {device}")
    model = whisperx.load_model(args.model, device, compute_type=compute_type, language=args.language)

    diarizer = None
    if token:
        from whisperx.diarize import DiarizationPipeline

        diarizer = DiarizationPipeline(use_auth_token=token, device=device)

    align_models = {}  # language -> (model, metadata), loaded on first use
    done, skipped, failed = 0, 0, []
    start = time.time()

    for i, src in enumerate(files, 1):
        out_dir = args.out or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (src.stem + SUFFIX)
        if out.exists():
            print(f"[{i}/{len(files)}] skip {src.name} (transcript exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(files)}] {src.name}")
        t0 = time.time()
        try:
            audio = whisperx.load_audio(str(src))
            result = model.transcribe(audio, batch_size=args.batch_size)
            language = result.get("language", "en")
            segments = result["segments"]

            if segments:
                if language not in align_models:
                    align_models[language] = whisperx.load_align_model(language_code=language, device=device)
                align_model, metadata = align_models[language]
                result = whisperx.align(segments, align_model, metadata, audio, device, return_char_alignments=False)
                if diarizer:
                    result = whisperx.assign_word_speakers(diarizer(audio), result)
                segments = result["segments"]

            write_transcript(out, src, language, average_confidence(segments), time.time() - t0, segments)
            done += 1
        except Exception as e:  # keep going; report at the end
            print(f"  failed: {e}", file=sys.stderr)
            failed.append(src.name)

    print(f"\n{done} done, {skipped} skipped, {len(failed)} failed in {time.time() - start:.0f} s")
    for name in failed:
        print(f"  failed: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
