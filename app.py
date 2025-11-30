import os
import re
import warnings
from typing import Type, List, Dict

import numpy as np
import gradio as gr
import PyPDF2
import faiss
import requests
import torch

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from huggingface_hub import login
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

warnings.filterwarnings("ignore")

# ===================== 2) Load FitMate model (4-bit if possible) =====================

print("⏳ Loading FitMate model on GPU if available...")

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please set it in your environment variables.")
login(token=HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained("moamenshamed/fitmate", token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
try:
    # Try loading 4-bit for speed
    print("Trying to load model in 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        "moamenshamed/fitmate",
        token=HF_TOKEN,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        torch_dtype=torch.bfloat16,
    )
    print("✅ Loaded in 4-bit mode.")
except Exception as e:
    print("⚠️ 4-bit load failed, falling back to standard fp16/fp32. Reason:", e)
    model = AutoModelForCausalLM.from_pretrained(
        "moamenshamed/fitmate",
        token=HF_TOKEN,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )

model.eval()

print(f"✅ FitMate loaded on: {device}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print()

# ===================== 3) System Instructions =====================

offtopic_rule = (
    "If the user asks about anything outside fitness, workouts, nutrition, health, "
    "or healthy lifestyle, politely decline in ONE short sentence and vary the wording "
    "each time (do not repeat the same sentence). After declining, briefly invite a "
    "fitness-related question. Do not mention policies or rules."
)

BASE_SYSTEM_PROMPT = (
    "You are FitMate, a highly knowledgeable AI assistant specializing ONLY in fitness, "
    "workouts, nutrition, health, recovery and lifestyle improvement.\n\n"
    "CORE RULES:\n"
    "• Stay strictly within fitness, health, nutrition, mindset, sleep and healthy lifestyle.\n"
    "• If the user asks about anything outside these topics, politely decline by saying you can "
    "only discuss fitness and healthy living. " + offtopic_rule + "\n"
    "• Never mention that you are following rules or policies.\n\n"
    "COACHING STYLE:\n"
    "• Think like an experienced personal trainer + nutrition coach.\n"
    "• Be detailed, motivating and practical (no fluff, always actionable).\n"
    "• For goals like weight loss, muscle gain or full programs, structure your "
    "response into clear sections (Assessment, Nutrition Plan, Workout Plan, Recovery, Tracking).\n"
    "• Ask for missing information you need (weight, height, age, activity level, equipment, injuries).\n"
    "• Adapt your advice to the user's goal: fat loss, muscle gain, strength, general health, performance, etc.\n"
    "• Give complete answers, not overly short summaries.\n"
    "• Always finish your answer with a short, friendly follow-up question inviting the user to continue.\n"
    "• Never cut off mid-sentence; complete your thought before stopping.\n\n"
    "LANGUAGE:\n"
    "• Detect the user's language automatically.\n"
    "• If they write mainly in Arabic, answer fully in Arabic (Egyptian dialect is fine) unless a term is better in English.\n"
    "• If they write mainly in English, answer in English.\n"
    "• If they mix Arabic and English, you may mix the languages naturally.\n"
)

# ===================== 4) Tools (Web Search + Scrape) =====================

class WebSearchToolInput(BaseModel):
    query: str = Field(..., description="Search query")

class WebSearchTool:
    name: str = "web_search"
    description: str = "Search the web using DuckDuckGo for current health and fitness information"
    args_schema: Type[BaseModel] = WebSearchToolInput
    
    def run(self, query: str) -> str:
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=5)
            if not results:
                return "No web results found."
            summary = f"Web search results for: {query}\n\n"
            for i, r in enumerate(results, 1):
                summary += f"{i}. {r['title']}\n   {r['body']}\n   {r['href']}\n\n"
            return summary[:3000]
        except Exception as e:
            return f"Search error: {str(e)}"

class ScrapeWebsiteToolInput(BaseModel):
    url: str = Field(..., description="URL to scrape")

class ScrapeWebsiteTool:
    name: str = "scrape_website"
    description: str = "Scrape content from a website URL"
    args_schema: Type[BaseModel] = ScrapeWebsiteToolInput
    
    def run(self, url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            content = "\n".join([p.get_text() for p in paragraphs[:10]])
            if not content.strip():
                return "No readable text content found on this page."
            return f"Website Content (first paragraphs):\n\n{content[:3000]}"
        except Exception as e:
            return f"Scraping error: {str(e)}"

web_search_tool = WebSearchTool()
scrape_website_tool = ScrapeWebsiteTool()

# ===================== 5) RAG: FAISS Vector Store =====================

class FAISSVectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks: List[str] = []
        self.metadata: List[dict] = []
    
    def add_pdf(self, pdf_path: str):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_path)
            chunks_data = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    chunk_size = 500
                    for i in range(0, len(text), chunk_size):
                        chunk_text = text[i : i + chunk_size].strip()
                        if len(chunk_text) > 100:
                            chunks_data.append({"text": chunk_text, "page": page_num + 1})
            
            if not chunks_data:
                return "❌ No text extracted from PDF"
            
            texts = [c["text"] for c in chunks_data]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings).astype("float32")
            
            self.index.add(embeddings)
            self.chunks.extend(texts)
            self.metadata.extend([{"page": c["page"]} for c in chunks_data])
            
            return f"✅ Loaded {len(chunks_data)} chunks from {len(pdf_reader.pages)} pages"
        except Exception as e:
            return f"❌ Error processing PDF: {str(e)}"
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        if len(self.chunks) == 0:
            return ""
        
        query_emb = self.embedding_model.encode([query])
        query_emb = np.array(query_emb).astype("float32")
        distances, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks) and dist < 1.5:
                results.append(f"[Page {self.metadata[idx]['page']}] {self.chunks[idx]}")
        
        return "\n\n".join(results) if results else ""
    
    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.metadata = []

print("Initializing vector store...")
vector_store = FAISSVectorStore()
print("✅ Vector store ready!\n")

# ===================== 6) Generation – Faster Settings =====================

def clean_fitmate_output(text: str) -> str:
    original = text
    lower = text.lower()

    idx = lower.rfind("final")
    if idx != -1 and idx + len("final") < len(text):
        text = text[idx + len("final"):]

    text = text.lstrip(" \n:-")
    if text.lower().startswith("analysis"):
        dot_idx = text.find(".")
        if dot_idx != -1 and dot_idx + 1 < len(text):
            text = text[dot_idx + 1:]

    text = re.sub(r"^user says[:\s\"\u201c]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^the user says[:\s\"\u201c]+", "", text, flags=re.IGNORECASE)

    text = text.strip()
    return text if text else original.strip()

def generate_with_fitmate_chat(
    user_content: str,
    max_new_tokens_cap: int = 1024,
    temperature: float = 0.4,
) -> str:
    system_short = BASE_SYSTEM_PROMPT[:1500]

    messages = [
        {"role": "system", "content": system_short},
        {"role": "user", "content": user_content},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_ids = inputs["input_ids"]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id

    context_window = int(getattr(model.config, "max_position_embeddings", 4096))
    input_length = int(input_ids.shape[1])
    available = context_window - input_length - 32
    if available <= 0:
        raise ValueError(
            f"No room left to generate tokens (input={input_length}, window={context_window}). "
            "Try shortening the conversation or question."
        )

    max_new_tokens = min(max_new_tokens_cap, available)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3,
            use_cache=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    generated_ids = outputs[0][input_length:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    answer_text = clean_fitmate_output(answer_text)
    return answer_text

# ===================== 7) Agent Logic + Language Detection + LoRA-mode =====================

def detect_language(text: str) -> str:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    if arabic_chars > latin_chars and arabic_chars > 0:
        return "arabic"
    if latin_chars > arabic_chars and latin_chars > 0:
        return "english"
    return "mixed"

def build_agent_user_message(
    question: str,
    doc_context: str,
    web_info: str,
    lora_mode: str,
) -> str:
    parts = []

    lang = detect_language(question)
    if lang == "arabic":
        parts.append(
            "The user is writing mainly in Arabic. Answer ONLY in Arabic (Egyptian dialect is fine), "
            "unless a specific term is clearly better in English.\n\n"
        )
    elif lang == "english":
        parts.append(
            "The user is writing mainly in English. Answer in English.\n\n"
        )
    else:
        parts.append(
            "The user is mixing Arabic and English. You may mix both languages naturally.\n\n"
        )

    if lora_mode and "Fat-loss" in lora_mode:
        parts.append("Coaching mode: focus on safe but effective fat-loss, calorie deficit, and adherence.\n\n")
    elif lora_mode and "Muscle" in lora_mode:
        parts.append("Coaching mode: focus on hypertrophy, strength, progressive overload, and high-protein nutrition.\n\n")
    elif lora_mode and "Rehab" in lora_mode:
        parts.append("Coaching mode: focus on gentle exercises, mobility, pain-free range of motion, and recovery.\n\n")

    if doc_context:
        parts.append(
            "Below is some DOCUMENT context from the user's fitness/nutrition PDFs. "
            "Use it only if it clearly helps answer the question:\n\n"
            f"{doc_context}\n\n"
        )

    if web_info:
        parts.append(
            "Below is some WEB SEARCH information about fitness/health that may be useful. "
            "Use it as supportive evidence only, and DO NOT mention that it came from the web:\n\n"
            f"{web_info}\n\n"
        )

    parts.append(
        "Now answer the user's question directly, speaking to them as their fitness coach. "
        "Do not talk about 'analysis' or 'steps', just give the advice itself.\n\n"
        f"User question: {question}\n\n"
        "Answer now in a clear, well-structured, motivational way."
    )

    return "".join(parts)

def run_agent(
    question: str,
    enable_web: bool,
    max_tokens: int,
    lora_mode: str,
) -> str:
    try:
        doc_context = vector_store.get_context(question)

        lower_q = question.lower()
        need_web = (
            enable_web
            and any(
                kw in lower_q
                for kw in [
                    "latest",
                    "recent",
                    "new study",
                    "new research",
                    "2023",
                    "2024",
                    "أحدث",
                    "آخر أبحاث",
                    "دراسة جديدة",
                ]
            )
        )

        web_info = ""
        if need_web:
            web_info = web_search_tool.run(question)

        user_msg = build_agent_user_message(question, doc_context, web_info, lora_mode)
        answer = generate_with_fitmate_chat(user_msg, max_new_tokens_cap=max_tokens, temperature=0.4)
        return answer

    except Exception as e:
        return f"❌ Error inside agent: {str(e)}"

# ===================== 8) Multi-Chat & Helpers (messages format) =====================

def create_new_chat(chat_histories: Dict[str, List[Dict[str, str]]]):
    idx = len(chat_histories) + 1
    chat_id = f"Chat {idx}"
    chat_histories[chat_id] = []
    choices = list(chat_histories.keys())
    return chat_histories, chat_id, gr.update(choices=choices, value=chat_id), []

def switch_chat(selected_chat: str, chat_histories: Dict[str, List[Dict[str, str]]]):
    history = chat_histories.get(selected_chat, [])
    return history

def delete_chat(chat_histories, current_chat):
    if current_chat in chat_histories:
        chat_histories.pop(current_chat)
    if not chat_histories:
        chat_histories["Chat 1"] = []
        current_chat = "Chat 1"
    else:
        current_chat = list(chat_histories.keys())[0]
    choices = list(chat_histories.keys())
    return chat_histories, current_chat, gr.update(choices=choices, value=current_chat), chat_histories[current_chat]

def upload_pdf(pdf_file):
    if pdf_file is None:
        return "⚠️ No file uploaded"
    return vector_store.add_pdf(pdf_file)

def add_user_message(message, chat_histories, current_chat):
    if not message or not message.strip():
        return chat_histories, chat_histories.get(current_chat, [])
    history = chat_histories.get(current_chat, [])
    history.append({"role": "user", "content": message})
    chat_histories[current_chat] = history
    return chat_histories, history

def generate_bot_reply(enable_web, max_tokens, lora_mode, chat_histories, current_chat):
    history = chat_histories.get(current_chat, [])
    if not history:
        return chat_histories, history
    if history[-1]["role"] != "user":
        return chat_histories, history

    user_message = history[-1]["content"]
    bot_response = run_agent(user_message, enable_web, int(max_tokens), lora_mode)
    history.append({"role": "assistant", "content": bot_response})
    chat_histories[current_chat] = history
    return chat_histories, history

def clear_current_chat(chat_histories, current_chat):
    chat_histories[current_chat] = []
    return chat_histories, []

def clear_documents():
    vector_store.clear()
    return "✅ All documents cleared from memory"

def update_account_name(new_name: str):
    new_name = (new_name or "").strip()
    if not new_name:
        new_name = "Guest Account"
    return new_name, gr.update(value=f"<div class='account-name'>👤 {new_name}</div>")

def rename_chat(new_name: str, chat_histories, current_chat):
    new_name = (new_name or "").strip()
    if not new_name or new_name == current_chat:
        return chat_histories, current_chat, gr.update(), gr.update(value=current_chat)
    if new_name in chat_histories:
        return chat_histories, current_chat, gr.update(), gr.update(value=current_chat)

    chat_histories[new_name] = chat_histories.pop(current_chat)
    current_chat = new_name
    choices = list(chat_histories.keys())
    return chat_histories, current_chat, gr.update(choices=choices, value=current_chat), gr.update(value=current_chat)

def set_chat_name_box(chat_id: str):
    return chat_id

# ===================== 9) Gradio Interface – Clean Chat UI =====================

with gr.Blocks(title="FitMate - AI Fitness Coach") as demo:
    gr.HTML("""
    <style>
        /* Global Reset & Deep Purple Theme */
        body, .gradio-container {
            background: linear-gradient(135deg, #0f0518 0%, #1a0b2e 100%) !important;
            color: #e9d5ff !important;
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            margin: 0;
            padding: 0;
        }
        
        /* Sidebar Styling */
        .sidebar-container {
            background-color: #130725; /* Darker purple */
            border-right: 1px solid #2d1b4e;
            padding: 24px;
            height: 100vh;
            display: flex;
            flex-direction: column;
            box-shadow: 4px 0 24px rgba(0,0,0,0.4);
        }
        
        .app-logo {
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }
        
        .app-subtitle {
            font-size: 0.85rem;
            color: #9ca3af;
            margin-bottom: 16px;
            font-weight: 300;
        }

        .pro-badge {
            background: linear-gradient(90deg, #7c3aed, #db2777);
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
            text-align: center;
            width: fit-content;
        }
        
        .account-display {
            background: rgba(255, 255, 255, 0.05);
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 20px;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }

        /* Main Panel Styling */
        .main-panel {
            padding: 0;
            height: 100vh;
            overflow-y: auto;
            background: transparent;
        }
        
        .header-section {
            padding: 20px 40px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            background: rgba(15, 5, 24, 0.5);
            backdrop-filter: blur(10px);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-pill {
            background: rgba(124, 58, 237, 0.15);
            padding: 6px 16px;
            border-radius: 999px;
            font-size: 0.8rem;
            color: #d8b4fe;
            border: 1px solid rgba(124, 58, 237, 0.3);
            font-weight: 500;
        }

        /* Tabs Styling */
        .tabs {
            margin-top: 0px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .tab-nav {
            border: none !important;
            background: transparent !important;
        }
        .tab-nav button {
            color: #9ca3af !important;
            font-weight: 500;
        }
        .tab-nav button.selected {
            border-bottom: 2px solid #d8b4fe !important;
            color: #d8b4fe !important;
            background: transparent !important;
            font-weight: 700;
        }

        /* Chatbot Styling */
        .bubble-wrap {
            background-color: #1e1035 !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        /* Input Area */
        .input-box textarea {
            background-color: #1e1035 !important;
            border: 1px solid rgba(124, 58, 237, 0.3) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 12px !important;
        }
        .input-box textarea:focus {
            border-color: #a78bfa !important;
            box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2) !important;
        }

        /* Buttons */
        button.primary {
            background: linear-gradient(90deg, #7c3aed, #6d28d9) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s;
        }
        button.primary:hover {
            box-shadow: 0 0 15px rgba(124, 58, 237, 0.5) !important;
            transform: translateY(-1px);
        }
        
        button.secondary {
            background-color: rgba(255,255,255,0.05) !important;
            color: #e9d5ff !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 8px !important;
        }

        /* Landing Page Hero */
        .hero-container {
            text-align: center;
            padding: 60px 20px;
            max-width: 800px;
            margin: 0 auto;
        }
        .hero-icon {
            font-size: 4rem;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 16px;
            color: white;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: #9ca3af;
            line-height: 1.6;
            margin-bottom: 40px;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }
        .feature-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            transition: transform 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.05);
            border-color: rgba(124, 58, 237, 0.3);
        }

        /* Hide footer */
        footer { display: none !important; }
    </style>
    """)

    chat_histories = gr.State({"Chat 1": []})
    current_chat = gr.State("Chat 1")
    account_name_state = gr.State("Guest Account")

    with gr.Row():
        # ========= Sidebar (Left) =========
        with gr.Column(scale=1, min_width=280):
            gr.Markdown(
                """
                <div class="app-logo">FitMate</div>
                <div class="app-subtitle">AI Fitness Coach</div>
                <div class="pro-badge">🏆 Pro Plan Unlimited</div>
                """
            )
            
            account_md = gr.Markdown(
                "<div class='account-display'>👤 Guest Account</div>"
            )
            
            new_chat_btn = gr.Button("+ New Chat", variant="primary")
            
            gr.Markdown("### Your Chats")
            
            # Chat list (Radio)
            chat_selector = gr.Radio(
                choices=["Chat 1"],
                value="Chat 1",
                label="",
                interactive=True
            )
            
            gr.Markdown("---")
            delete_chat_btn = gr.Button("🗑️ Delete Chat", variant="secondary")

        # ========= Main Panel (Right) =========
        with gr.Column(scale=4):
            
            # Header
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown(
                        """
                        # Dashboard
                        """
                    )
                with gr.Column(scale=1):
                     gr.HTML(
                        """
                        <div style="display:flex; gap:8px; justify-content:flex-end; margin-top: 10px;">
                            <span class="header-pill">🔥 Pro Active</span>
                            <span class="header-pill">⚡ Turbo Mode</span>
                        </div>
                        """
                    )

            with gr.Tabs():
                
                # 1. Overview Tab (Landing Page Style)
                with gr.Tab("🏠 Home"):
                    gr.HTML(
                        """
                        <div class="hero-container">
                            <div class="hero-icon">💪</div>
                            <h1 class="hero-title">Transform Your Fitness Journey</h1>
                            <p class="hero-subtitle">
                                Your personal AI coach for custom workouts, nutrition plans, and recovery advice. 
                                Powered by advanced AI to guide you step-by-step.
                            </p>
                            
                            <div class="feature-grid">
                                <div class="feature-card">
                                    <div style="font-size:2rem; margin-bottom:10px;">🏋️</div>
                                    <div style="font-weight:bold; color:white;">Smart Workouts</div>
                                    <div style="font-size:0.8rem; color:#9ca3af;">Personalized routines</div>
                                </div>
                                <div class="feature-card">
                                    <div style="font-size:2rem; margin-bottom:10px;">🥗</div>
                                    <div style="font-weight:bold; color:white;">Nutrition Plans</div>
                                    <div style="font-size:0.8rem; color:#9ca3af;">Macros & Meal Prep</div>
                                </div>
                                <div class="feature-card">
                                    <div style="font-size:2rem; margin-bottom:10px;">🧠</div>
                                    <div style="font-weight:bold; color:white;">Expert Knowledge</div>
                                    <div style="font-size:0.8rem; color:#9ca3af;">RAG & Web Search</div>
                                </div>
                                <div class="feature-card">
                                    <div style="font-size:2rem; margin-bottom:10px;">⚡</div>
                                    <div style="font-weight:bold; color:white;">Instant Answers</div>
                                    <div style="font-size:0.8rem; color:#9ca3af;">24/7 Availability</div>
                                </div>
                            </div>
                            
                            <div style="margin-top: 40px; padding: 20px; background: rgba(124, 58, 237, 0.1); border-radius: 12px; border: 1px solid rgba(124, 58, 237, 0.3);">
                                <div style="font-weight:bold; color:#d8b4fe; margin-bottom:8px;">🚀 Ready to start?</div>
                                <div style="color:#e9d5ff; font-size:0.9rem;">Switch to the <b>Chat Tab</b> to begin your session.</div>
                            </div>
                        </div>
                        """
                    )

                # 2. Chat Tab
                with gr.Tab("💬 Chat"):
                    chatbot = gr.Chatbot(
                        label="FitMate Pro",
                        height=600
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask FitMate anything...",
                            show_label=False,
                            lines=3,
                            scale=8
                        )
                        with gr.Column(scale=1):
                            send_btn = gr.Button("Send ➤", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")

                # 3. Documents Tab
                with gr.Tab("📄 Knowledge Base"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📂 Upload Context")
                            gr.Markdown("Enhance your AI coach with personal documents (PDFs).")
                            
                            pdf_upload = gr.File(
                                label="Upload PDF",
                                file_types=[".pdf"],
                                file_count="single"
                            )
                            upload_btn = gr.Button("Process Document", variant="primary")
                            upload_status = gr.Textbox(label="Processing Status", interactive=False)
                            
                            gr.Markdown("---")
                            clear_docs_btn = gr.Button("🗑️ Clear Memory", variant="stop")
                            clear_docs_status = gr.Textbox(label="Memory Status", interactive=False)

                # 4. Settings Tab
                with gr.Tab("⚙️ Settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 👤 User Profile")
                            display_name_box = gr.Textbox(label="Display Name", value="Guest Account")
                            
                            gr.Markdown("### 💬 Session Management")
                            chat_name_box = gr.Textbox(label="Rename Current Chat", value="Chat 1")
                            rename_chat_btn = gr.Button("Update Name")
                            
                        with gr.Column():
                            gr.Markdown("### ⚡ Pro Configuration")
                            
                            # Visual "Plan" indicator
                            gr.HTML("""
                            <div style="background: rgba(124, 58, 237, 0.1); padding: 15px; border-radius: 8px; border: 1px solid rgba(124, 58, 237, 0.3); margin-bottom: 20px;">
                                <div style="font-weight: bold; color: #d8b4fe;">Current Plan: Pro Unlimited</div>
                                <div style="font-size: 0.8rem; color: #a78bfa;">You have access to maximum context window and web search capabilities.</div>
                            </div>
                            """)
                            
                            lora_mode = gr.Dropdown(
                                choices=[
                                    "Standard coach (default)",
                                    "Fat-loss focused",
                                    "Muscle-building focused",
                                    "Rehab & mobility focused",
                                ],
                                value="Standard coach (default)",
                                label="Coaching Focus"
                            )
                            
                            enable_web = gr.Checkbox(
                                value=True,
                                label="Enable Web Search (Pro Feature)"
                            )
                            
                            max_tokens_slider = gr.Slider(
                                minimum=256,
                                maximum=2048,
                                value=2048, # Set to max for "Pro" feel
                                step=64,
                                label="Response Length (Pro Limit: 2048)"
                            )

    # ---------- Wiring: New Chat ----------
    new_chat_btn.click(
        create_new_chat,
        inputs=[chat_histories],
        outputs=[chat_histories, current_chat, chat_selector, chatbot],
    ).then(
        set_chat_name_box,
        inputs=[current_chat],
        outputs=[chat_name_box],
    )

    # ---------- Wiring: Switch Chat ----------
    chat_selector.change(
        switch_chat,
        inputs=[chat_selector, chat_histories],
        outputs=[chatbot],
    ).then(
        lambda selected: selected,
        inputs=[chat_selector],
        outputs=[current_chat],
    ).then(
        set_chat_name_box,
        inputs=[chat_selector],
        outputs=[chat_name_box],
    )

    # ---------- Wiring: Delete Chat ----------
    delete_chat_btn.click(
        delete_chat,
        inputs=[chat_histories, current_chat],
        outputs=[chat_histories, current_chat, chat_selector, chatbot],
    ).then(
        set_chat_name_box,
        inputs=[current_chat],
        outputs=[chat_name_box],
    )

    # ---------- Wiring: Chat ----------
    send_btn.click(
        add_user_message,
        inputs=[msg, chat_histories, current_chat],
        outputs=[chat_histories, chatbot],
    ).then(
        generate_bot_reply,
        inputs=[enable_web, max_tokens_slider, lora_mode, chat_histories, current_chat],
        outputs=[chat_histories, chatbot],
    ).then(
        lambda: "",
        inputs=None,
        outputs=msg,
    )

    msg.submit(
        add_user_message,
        inputs=[msg, chat_histories, current_chat],
        outputs=[chat_histories, chatbot],
    ).then(
        generate_bot_reply,
        inputs=[enable_web, max_tokens_slider, lora_mode, chat_histories, current_chat],
        outputs=[chat_histories, chatbot],
    ).then(
        lambda: "",
        inputs=None,
        outputs=msg,
    )

    clear_btn.click(
        clear_current_chat,
        inputs=[chat_histories, current_chat],
        outputs=[chat_histories, chatbot],
    )

    # ---------- Wiring: Documents ----------
    upload_btn.click(upload_pdf, [pdf_upload], [upload_status])
    clear_docs_btn.click(clear_documents, None, [clear_docs_status])

    # ---------- Wiring: Account name ----------
    display_name_box.change(
        update_account_name,
        inputs=[display_name_box],
        outputs=[account_name_state, account_md],
    )

    # ---------- Wiring: Rename chat ----------
    rename_chat_btn.click(
        rename_chat,
        inputs=[chat_name_box, chat_histories, current_chat],
        outputs=[chat_histories, current_chat, chat_selector, chat_name_box],
    )

if __name__ == "__main__":
    demo.launch(share=True)
