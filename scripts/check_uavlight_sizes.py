from huggingface_hub import HfApi

api = HfApi()
info = api.repo_info(repo_id="dukang92/UAVLight", repo_type="dataset", files_metadata=True)
print("=== UAVLight Dataset Files ===")
for s in info.siblings:
    size_mb = (s.size or 0) / (1024**2)
    print(f"{s.rfilename:35s} {size_mb:8.2f} MB")
