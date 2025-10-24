import re
import json
import jwt
import requests as r
from time import time, sleep
from cachetools import cached, TTLCache
from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

@cached(cache=TTLCache(maxsize=1024, ttl=3600))
def get_ya_token(private_key: str, service_id: str, key_id: str):
    now = int(time())
    payload = {
        "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
        "iss": service_id,
        "iat": now,
        "exp": now + 360,
    }
    
    encoded_token = jwt.encode(
        payload, private_key, algorithm="PS256", headers={"kid": key_id}
    )
    
    iam_token = r.post(
        "https://iam.api.cloud.yandex.net/iam/v1/tokens", json={"jwt": encoded_token}
    )
    if iam_token.status_code != 200:
        raise Exception("Wrong IAM token response")
    return iam_token.json()["iamToken"]


def load_system_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    return system_prompt


def format_messages(context, question):
    return [
        {"role": "system", "text": load_system_prompt()},
        {"role": "user", "text": f"Контекст: {context}\n Вопрос: {question}"},
    ]

def get_answer(context, question, settings):

    client = {"token": get_ya_token(settings.PRIVATE_KEY, settings.SERVICE_ACCOUNT_ID, settings.KEY_ID), "folder_id": "b1ggivrnbg1ftsr8no1s"}
    
    values = {
        "modelUri": "gpt://b1ggivrnbg1ftsr8no1s/yandexgpt-lite/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": str(settings.LLM_MAX_OUTPUT)
        },
        "messages": format_messages(context, question)
    }
    
    resp = r.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion", 
        json=values, 
        headers={"Authorization": f"Bearer {client['token']}", "x-folder-id": "b1ggivrnbg1ftsr8no1s"}
    )
    
    if resp.status_code != 200:
        raise Exception(f"Yagpt error: {resp.text}")
        
    response_data = resp.json()
    answer = response_data['result']['alternatives'][0]['message']['text']
    
    return answer + '\n' + settings.warning_message