from pathlib import Path
import whisper
import csv

ROOT = Path(__file__).resolve().parent


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    return ROOT / "melo" / path

def cer(ref, hyp):
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")

    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n] / max(1, m)

manifest = ROOT / "outputs/raiden_test_generated_G8000/manifest.list"
out_csv = ROOT / "outputs/raiden_test_generated_G8000/cer_result.csv"

model = whisper.load_model("small")

rows = []
for idx, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    ref_wav, gen_wav, ref_text = line.split("|", 2)
    gen_wav_path = resolve_repo_path(gen_wav)
    result = model.transcribe(str(gen_wav_path), language="zh", fp16=False)
    hyp_text = result["text"].strip()
    score = cer(ref_text, hyp_text)

    rows.append({
        "id": idx,
        "generated_wav": gen_wav,
        "ref_text": ref_text,
        "asr_text": hyp_text,
        "cer": score,
    })

avg_cer = sum(r["cer"] for r in rows) / len(rows)

with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "generated_wav", "ref_text", "asr_text", "cer"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Average CER: {avg_cer:.4f}")
print(f"Saved to: {out_csv}")
