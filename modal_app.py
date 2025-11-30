import modal

app = modal.App("fitmate")

# Install all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "gradio",
        "transformers",
        "accelerate", 
        "bitsandbytes",
        "sentence-transformers",
        "faiss-cpu",
        "pypdf2",
        "duckduckgo-search",
        "beautifulsoup4",
        "requests",
        "huggingface_hub",
        "torch",
        "typing_extensions",
        "regex",
    )
)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.web_server(7860, startup_timeout=600)
def web():
    # Import and run the app
    import subprocess
    subprocess.Popen(["python", "/root/app.py"])
    
    # Keep the server running
    import time
    while True:
        time.sleep(60)
