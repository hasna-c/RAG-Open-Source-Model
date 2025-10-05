# baseline.py
import requests
import time
import csv
import subprocess
import GPUtil

# ---------- CONFIG ----------
SERVER_URL = "http://127.0.0.1:8000/ask"  # your running RAG server
QUESTIONS = [
    "What is a waterfall model?",
    "What are the phases in SCRUM?",
    "What is the objective of prototyping?",
    "What are the stages in Testing?",
    "What is the importance of software engineering?"
]
CSV_FILE = "baseline_results.csv"

# ---------- GPU MONITORING FUNCTIONS ----------
def get_gpu_gputil():
    """Get GPU usage using GPUtil"""
    try:
        gpu = GPUtil.getGPUs()[0]  # first GPU
        return {
            "gpu_name": gpu.name,
            "mem_used_MB": gpu.memoryUsed,
            "mem_total_MB": gpu.memoryTotal,
            "gpu_load_pct": gpu.load * 100
        }
    except Exception as e:
        return None

def get_gpu_nvsmi():
    """Get GPU usage using nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True
        )
        used, total, load = map(int, result.stdout.strip().split(", "))
        return {
            "mem_used_MB": used,
            "mem_total_MB": total,
            "gpu_load_pct": load
        }
    except Exception as e:
        return None

# ---------- BASELINE TESTING ----------
results = []

for q in QUESTIONS:
    # Record GPU usage before inference
    gpu_gputil_start = get_gpu_gputil()
    gpu_nvsmi_start = get_gpu_nvsmi()

    start_time = time.time()
    # Send request to RAG server
    response = requests.post(SERVER_URL, json={"question": q})
    elapsed_time = time.time() - start_time

    # Record GPU usage after inference
    gpu_gputil_end = get_gpu_gputil()
    gpu_nvsmi_end = get_gpu_nvsmi()

    # Get answer
    if response.status_code == 200:
        answer = response.json().get("answer", "")
    else:
        answer = f"Error: {response.status_code}"

    # Append results
    results.append({
        "question": q,
        "answer": answer,
        "inference_time_sec": elapsed_time,
        # GPUtil stats
        "gputil_mem_start_MB": gpu_gputil_start.get("mem_used_MB") if gpu_gputil_start else None,
        "gputil_mem_end_MB": gpu_gputil_end.get("mem_used_MB") if gpu_gputil_end else None,
        "gputil_load_start_pct": gpu_gputil_start.get("gpu_load_pct") if gpu_gputil_start else None,
        "gputil_load_end_pct": gpu_gputil_end.get("gpu_load_pct") if gpu_gputil_end else None,
        # nvidia-smi stats
        "nvsmi_mem_start_MB": gpu_nvsmi_start.get("mem_used_MB") if gpu_nvsmi_start else None,
        "nvsmi_mem_end_MB": gpu_nvsmi_end.get("mem_used_MB") if gpu_nvsmi_end else None,
        "nvsmi_load_start_pct": gpu_nvsmi_start.get("gpu_load_pct") if gpu_nvsmi_start else None,
        "nvsmi_load_end_pct": gpu_nvsmi_end.get("gpu_load_pct") if gpu_nvsmi_end else None,
    })

    print(f"Answered: {q} | Time: {elapsed_time:.2f}s | GPUtil Mem: {gpu_gputil_start.get('mem_used_MB') if gpu_gputil_start else '-'} -> {gpu_gputil_end.get('mem_used_MB') if gpu_gputil_end else '-'} MB | nvidia-smi Mem: {gpu_nvsmi_start.get('mem_used_MB') if gpu_nvsmi_start else '-'} -> {gpu_nvsmi_end.get('mem_used_MB') if gpu_nvsmi_end else '-'} MB")

# ---------- SAVE TO CSV ----------
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Baseline results saved to {CSV_FILE}")
