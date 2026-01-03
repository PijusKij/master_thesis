import json
import re
import time
import pandas as pd
import os
import glob
import requests
from tqdm.auto import tqdm

#This code was used to run all three models - gpt-oss20b, medgemma 4b, gemma3 4b
#This code was used to run both prediction and prediction tasks, changing the prompt directory



# server setup 
# LLM_URL = "http://x/api/generate"


# Choose a model based on task
#MODEL_NAME_DEFAULT = "hf.co/unsloth/gpt-oss-20b-GGUF:latest"
#MODEL_NAME_DEFAULT = "hf.co/unsloth/gemma-3-4b-it-GGUF"
#MODEL_NAME_DEFAULT = "hf.co/unsloth/medgemma-4b-it-GGUF"
MODEL_NAME_ALT     = "other-model-name"

def call_hf_llm(prompt: str, model_name: str = MODEL_NAME_DEFAULT) -> str:
    """
    Send prompt to local LLM server and get raw completion text.

    `model_name` selects which model on the server to use.
    Adjust the JSON payload / response parsing to match your server's API.
    """
    payload = {
        "prompt": prompt,
        "model": model_name,   
        "max_tokens": 128,
        "temperature": 0.2,
        "stream": False
    }

    resp = requests.post(LLM_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()  
    raw = (
        data.get("text")
        or data.get("response")
        or data.get("generated_text")
        or data.get("output")
    )

    if raw is None:
        raise ValueError(f"Unexpected response format from LLM server: {data}")

    return str(raw).strip()


def parse_llm_output(raw: str, expected_prompt_id: int):
    """
    Parse {"pred_rmssd": float}. If JSON fails, grab first number.
    We still return (prompt_id, pred) so the rest of the pipeline works.
    """
    try:
        obj = json.loads(raw)
        yhat = float(obj["pred_rmssd"])
        return expected_prompt_id, yhat
    except Exception:
        m = re.search(r"[-+]?\d+(\.\d+)?", raw)
        if not m:
            raise ValueError(f"Could not parse prediction from: {raw}")
        yhat = float(m.group(0))
        return expected_prompt_id, yhat



BASE_DIR    = "/home/"
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")      # folder containing prompts_*.parquet
PRED_DIR    = os.path.join(BASE_DIR, "results")  # folder to save LLM predictions
os.makedirs(PRED_DIR, exist_ok=True)

N_SAMPLES = None  # set to None to use ALL prompts in each file

timing_rows = []

# Find all prompt files
prompt_files = sorted(
    glob.glob(os.path.join(PROMPTS_DIR, "*.parquet"))
)
for prompts_path in prompt_files:
    print(f"\n▶ Running LLM on prompt file: {prompts_path}")

    prompts_df = pd.read_parquet(prompts_path)

    work_df = prompts_df.head(N_SAMPLES) if N_SAMPLES is not None else prompts_df
    n_prompts = len(work_df)
    print(f"   Using {n_prompts} prompts from this file")

    pred_rows = []

    t0 = time.perf_counter()

    for _, row in tqdm(work_df.iterrows(), total=n_prompts, desc=os.path.basename(prompts_path)):
        pid = int(row["prompt_id"])
        prompt = row["prompt"]

        raw = call_hf_llm(prompt, model_name=MODEL_NAME_DEFAULT)

        out_pid, yhat = parse_llm_output(raw, expected_prompt_id=pid)

        if out_pid != pid:
            print(f"⚠️ ID mismatch: model {out_pid}, expected {pid}")

        pred_rows.append(
            {
                "prompt_id": pid,
                "user_id": row.get("user_id", None),
                "target_date": row.get("target_date", None),
                "y_true_rmssd": row.get("y_true_rmssd", None),
                "y_pred_rmssd": yhat,
            }
        )
    t1 = time.perf_counter()

    elapsed = t1 - t0
    time_per_prompt = elapsed / n_prompts if n_prompts > 0 else float("nan")
    throughput = n_prompts / elapsed if elapsed > 0 else float("nan")

  
    base_name = os.path.basename(prompts_path)
    preds_name = base_name.replace("prompts_", "preds_")
    preds_path = os.path.join(PRED_DIR, preds_name)

    preds_df = pd.DataFrame(pred_rows)
    preds_df.to_parquet(preds_path, index=False)

    print(f"   ⏱ elapsed={elapsed:.2f}s | per_prompt={time_per_prompt:.3f}s | throughput={throughput:.2f} prompts/s")
    print(f"   💾 Saved predictions to: {preds_path}")

    timing_rows.append(
        {
            "prompts_file": prompts_path,
            "preds_file": preds_path,
            "n_prompts": n_prompts,
            "elapsed_sec": elapsed,
            "time_per_prompt_sec": time_per_prompt,
            "throughput_prompts_per_sec": throughput,
        }
    )

timing_df = pd.DataFrame(timing_rows)
timing_df_path = os.path.join(PRED_DIR, "llm_timing_summary.csv")
timing_df.to_csv(timing_df_path, index=False)

print("\n✅ Finished all files.")
print(f"Timing summary saved to: {timing_df_path}")
