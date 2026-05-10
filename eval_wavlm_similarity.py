from pathlib import Path
import csv
import torch
import librosa
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMForXVector

ROOT = Path(__file__).resolve().parent


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    return ROOT / "melo" / path

manifest = ROOT / "outputs/raiden_test_generated_G8000/manifest.list"
out_csv = ROOT / "outputs/raiden_test_generated_G8000/speaker_similarity_result.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
model.eval()

def get_embedding(path):
    path = resolve_repo_path(path)
    wav, sr = librosa.load(path, sr=16000, mono=True)
    inputs = feature_extractor(
        wav,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        emb = model(**inputs).embeddings

    emb = F.normalize(emb, dim=-1)
    return emb

rows = []
for idx, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    ref_wav, gen_wav, text = line.split("|", 2)

    ref_emb = get_embedding(ref_wav)
    gen_emb = get_embedding(gen_wav)

    sim = F.cosine_similarity(ref_emb, gen_emb).item()

    rows.append({
        "id": idx,
        "reference_wav": ref_wav,
        "generated_wav": gen_wav,
        "similarity": sim,
    })

avg_sim = sum(r["similarity"] for r in rows) / len(rows)

with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "reference_wav", "generated_wav", "similarity"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Average speaker similarity: {avg_sim:.4f}")
print(f"Saved to: {out_csv}")
