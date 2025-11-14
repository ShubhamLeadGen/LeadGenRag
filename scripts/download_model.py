from sentence_transformers import SentenceTransformer
import os

# The model we need to download
model_name = "BAAI/bge-base-en-v1.5"

# The local directory where we will save it
save_path = os.path.join("models", model_name.replace("/", "_"))

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

print(f"Downloading model '{model_name}' to '{save_path}'...")

# Initialize the model from the Hub, which will download it
model = SentenceTransformer(model_name)

# Save the model to the specified path
model.save(save_path)

print(f"Model '{model_name}' downloaded successfully!")
