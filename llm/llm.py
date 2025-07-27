import os
from pathlib import Path
from gigachat import GigaChat
from gigachat.models import Chat, Messages

PROMPT_PATH = Path(__file__).parent / "prompt.txt"
CA_BUNDLE_PATH = Path(__file__).parent / "russian_trusted_root_ca.crt"

def get_giga_client(credentials):
    
    giga = GigaChat(
        credentials=credentials,
        ca_bundle_file=str(CA_BUNDLE_PATH),
        verify_ssl_certs=True
    )
    return giga
    

def load_system_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    return system_prompt


def format_messages(context, question):
    return [
        Messages(role="system", content=load_system_prompt()),
        Messages(role="user", content=f"Контекст: {context}\nВопрос: {question}"),
    ]

def get_answer(context, question, credentials, settings):
    giga = get_giga_client(credentials)
    
    chat = Chat(
        messages=format_messages(context, question),
        max_tokens=settings.GIGA_MAX_TOKENS,
        profanity_check=settings.PROFANITY_CHECK,
    )
    
    response = giga.chat(chat)
    return response.choices[0].message.content