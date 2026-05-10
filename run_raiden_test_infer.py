from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from melo.api import TTS

metadata = ROOT / "data/raiden_dataset/metadata_test.list"
out_dir = ROOT / "outputs/raiden_test_generated_G8000"
out_dir.mkdir(parents=True, exist_ok=True)

model = TTS(
    language="ZH",
    device="cuda:0",
    config_path=str(ROOT / "melo/logs/raiden_dataset/config.json"),
    ckpt_path=str(ROOT / "melo/logs/raiden_dataset/G_8000.pth"),
)

speaker_id = model.hps.data.spk2id["raiden"]

manifest_lines = []

for i, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
    ref_wav, spk, lang, text = line.split("|", 3)
    gen_wav = out_dir / f"test_{i:03d}.wav"

    print(f"{i:03d}: {text}")
    model.tts_to_file(text, speaker_id, str(gen_wav), speed=1.0)

    gen_wav_rel = gen_wav.relative_to(ROOT).as_posix()
    manifest_lines.append(f"{ref_wav}|{gen_wav_rel}|{text}")

(out_dir / "manifest.list").write_text("\n".join(manifest_lines), encoding="utf-8")
