import re
import json
import jwt
import requests as r
from time import time, sleep
from cachetools import cached, TTLCache
from pathlib import Path
import os

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def load_key_data():
    key_path = Path("/app/key.json")
    if key_path.exists():
        with open(key_path, "r") as f:
            return json.load(f)
    
    service_account_id = os.getenv("SERVICE_ACCOUNT_ID")
    private_key = os.getenv("PRIVATE_KEY")
    key_id = os.getenv("KEY_ID")
    
    if service_account_id and private_key and key_id:
        return {
            "service_account_id": service_account_id,
            "private_key": private_key,
            "id": key_id
        }
    
@cached(cache=TTLCache(maxsize=1024, ttl=3600))
def get_ya_token():
    key_data = load_key_data()
    now = int(time())
    payload = {
        "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
        "iss": key_data["service_account_id"],
        "iat": now,
        "exp": now + 360,
    }
    
    encoded_token = jwt.encode(
        payload, key_data["private_key"], algorithm="PS256", headers={"kid": key_data["id"]}
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
    client = {"token": get_ya_token(), "folder_id": "b1ggivrnbg1ftsr8no1s"}
    
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
