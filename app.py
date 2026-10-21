import base64

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="Mera AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Mera AI Chatbot")

TEXT_EXT = {"txt", "md", "csv", "json", "py", "js", "html", "css", "xml", "yaml", "yml", "log"}
IMG_EXT = {"png", "jpg", "jpeg", "webp"}
ALLOWED = sorted(TEXT_EXT | IMG_EXT | {"pdf"})
MAX_CHARS = 20000  # ek file se maximum itna text bheja jaayega

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Apni API Key daalein",
        type="password",
        placeholder="sk-...",
        help="OpenAI, OpenRouter, ya kisi bhi OpenAI-compatible API ki key",
    )
    base_url = st.text_input(
        "Base URL (optional)",
        value="https://api.openai.com/v1",
        help="OpenRouter ke liye: https://openrouter.ai/api/v1",
    )
    model_name = st.text_input("Model name", value="gpt-4o-mini")
    st.caption("Image samajhne ke liye vision model chahiye (jaise gpt-4o-mini).")
    st.caption("API key sirf aapke session mein rahegi, save nahi hoti.")
    if st.button("Chat clear karein"):
        st.session_state.pop("messages", None)
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! Main aapki kaise madad kar sakta hoon? Aap file bhi attach kar sakte hain.",
            "files": [],
            "api_content": "Namaste! Main aapki kaise madad kar sakta hoon? Aap file bhi attach kar sakte hain.",
        }
    ]


def read_file(f):
    """Ek uploaded file ko text ya image data mein badalta hai."""
    ext = f.name.rsplit(".", 1)[-1].lower()
    if ext in IMG_EXT:
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        b64 = base64.b64encode(f.getvalue()).decode()
        return {"type": "image", "name": f.name, "url": f"data:{mime};base64,{b64}"}
    if ext == "pdf":
        reader = PdfReader(f)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    else:
        text = f.getvalue().decode("utf-8", errors="ignore")
    return {"type": "text", "name": f.name, "text": text[:MAX_CHARS]}


def build_api_content(prompt, parsed):
    """Prompt + files ko API ke format mein jodta hai."""
    text = prompt
    for p in parsed:
        if p["type"] == "text":
            text += f"\n\n--- File: {p['name']} ---\n{p['text']}"
    images = [p for p in parsed if p["type"] == "image"]
    if not images:
        return text
    parts = [{"type": "text", "text": text}]
    for img in images:
        parts.append({"type": "image_url", "image_url": {"url": img["url"]}})
    return parts


# Purani chat dikhayein
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["files"]:
            st.caption("📎 " + ", ".join(msg["files"]))

# Chat input (file attach ke saath)
user_input = st.chat_input(
    "Apna message likhein...",
    accept_file="multiple",
    file_type=ALLOWED,
)

if user_input:
    prompt = (user_input.text or "").strip()
    files = list(user_input.files or [])

    if not api_key:
        st.info("Pehle sidebar mein apni API key daalein.")
        st.stop()
    if not prompt and not files:
        st.stop()
    if not prompt:
        prompt = "Is file ko dekhkar batayein ismein kya hai."

    parsed = []
    for f in files:
        try:
            parsed.append(read_file(f))
        except Exception as e:
            st.warning(f"{f.name} padh nahi paaye: {e}")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "files": [p["name"] for p in parsed],
            "api_content": build_api_content(prompt, parsed),
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)
        if parsed:
            st.caption("📎 " + ", ".join(p["name"] for p in parsed))
        for p in parsed:
            if p["type"] == "image":
                st.image(p["url"], width=240)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        api_messages = [
            {"role": m["role"], "content": m["api_content"]}
            for m in st.session_state.messages
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=api_messages,
            stream=True,
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "files": [],
                "api_content": full_response,
            }
        )

    except Exception as e:
        st.session_state.messages.pop()  # fail hui request history se hata dein
        st.error(f"Error: {e}")
