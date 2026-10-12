import kagglehub

# Download latest version
path = kagglehub.dataset_download("alexanderfrosati/goodbooks-10k-updated")

print("Path to dataset files:", path)
