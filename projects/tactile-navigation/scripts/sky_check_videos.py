"""Decode saved videos and produce start/middle/end sheets for visual review."""
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "outputs/sky_results.json").read_text())
    results = []
    for row in data["evaluation_records"]:
        video = (ROOT / row["record"]).with_name(Path(row["record"]).name.replace(".eval.json", ".mp4"))
        reader = imageio.get_reader(str(video), input_params=["-threads", "2"])
        meta = reader.get_meta_data()
        expected = round(meta["duration"] * meta["fps"])
        targets = {0, expected // 2, max(0, expected - 2)}
        samples = {}
        frames = 0
        for i, frame in enumerate(reader):
            if i in targets:
                samples[i] = Image.fromarray(frame)
            frames = i + 1
        reader.close()
        assert frames > 0 and abs(frames - expected) <= 2
        assert len(samples) == 3
        sheet = Image.new("RGB", (1920, 520), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (i, im) in enumerate(sorted(samples.items())):
            im.thumbnail((640, 470))
            sheet.paste(im, (column * 640, 35))
            draw.text((column * 640 + 10, 10), f"{row['room']} | t={i/meta['fps']:.1f}s | simulation replay", fill="black")
        sheet_path = ROOT / "outputs" / f"{row['room']}_video_review.jpg"
        sheet.save(sheet_path, quality=92)
        results.append(dict(room=row["room"], video=str(video.relative_to(ROOT)), decoded_frames=frames,
                            fps=meta["fps"], duration_s=meta["duration"], bytes=video.stat().st_size,
                            contact_sheet=str(sheet_path.relative_to(ROOT))))
    (ROOT / "outputs/sky_video_validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
