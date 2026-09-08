from huggingface_hub import HfApi, list_repo_files

api = HfApi()
try:
    files = list_repo_files(repo_id="dukang92/UAVLight", repo_type="dataset")
    print("[SUCCESS] Found files in dukang92/UAVLight:")
    for f in files[:20]:
        print(" -", f)
except Exception as e:
    print("[ERROR]", e)
