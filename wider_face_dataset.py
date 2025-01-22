import kagglehub

# Download latest version
path = kagglehub.dataset_download("iamprateek/wider-face-a-face-detection-dataset")

print("Path to dataset files:", path)
