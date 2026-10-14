"""Process both fisheye video tracks in a DJI Osmo 360 OSV recording."""
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
from pathlib import Path


def _video_streams(source: Path) -> list[dict]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,width,height,avg_frame_rate",
         "-of", "json", str(source)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    lenses = [stream for stream in streams
              if stream.get("codec_type") == "video"
              and stream.get("codec_name") == "hevc"
              and stream.get("width", 0) >= 1000
              and stream.get("height", 0) >= 1000]
    if len(lenses) != 2:
        raise ValueError(
            f"Expected two full-resolution HEVC lens streams in OSV, found {len(lenses)}. "
            "This recording needs inspection before processing."
        )
    return sorted(lenses, key=lambda stream: stream["index"])


def predict_osv(args, model_factory=None) -> int:
    """Accept OSV directly and produce one annotated MP4 per fisheye lens.

    Temporary, lossless remuxed streams are internal implementation details;
    the original OSV and its camera metadata are never modified.
    """
    from .video import predict_video

    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    streams = _video_streams(source)
    output = args.output.resolve() / args.name
    if output.exists() and not args.exist_ok and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}; choose --name or use --exist-ok")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    with tempfile.TemporaryDirectory(prefix=".osv_tracks_", dir=output) as temporary:
        for lens_number, stream in enumerate(streams):
            lens = f"lens{lens_number}"
            track = Path(temporary) / f"{lens}.mp4"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
                 "-map", f"0:{stream['index']}", "-an", "-sn", "-dn", "-c:v", "copy",
                 "-f", "mp4", str(track)],
                check=True, capture_output=True,
            )
            lens_args = copy.copy(args)
            lens_args.source = track
            lens_args.output = output
            lens_args.name = lens
            predict_video(lens_args, model_factory=model_factory)
            lens_dir = output / lens
            summary = json.loads((lens_dir / "summary.json").read_text(encoding="utf-8"))
            for key, filename in (
                ("video", f"{lens}_detected.mp4"),
                ("records", f"{lens}_detections.jsonl"),
            ):
                Path(summary[key]).replace(output / filename)
                summary[key] = str(output / filename)
            summary["input"] = str(source)
            summary["source_stream_index"] = stream["index"]
            summary["lens"] = lens
            (output / f"{lens}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (lens_dir / "summary.json").unlink()
            lens_dir.rmdir()
            summaries.append(summary)
            track.unlink()
    aggregate = {
        "input": str(source),
        "format": "DJI OSV dual-fisheye",
        "projection": "two unstitched fisheye views",
        "lenses": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"input": str(source), "output": str(output),
                      "videos": [item["video"] for item in summaries]}, ensure_ascii=False))
    return 0
