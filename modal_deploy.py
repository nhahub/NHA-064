import modal

app = modal.App("fitmate")

# Install dependencies
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
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.append("/root")
    
    # Import your app
    from app import demo
    
    # Return the Gradio app
    return demo.launch(prevent_thread_lock=True).app
