import re
import json
import jwt
import requests as r
from time import time, sleep
from cachetools import cached, TTLCache


@cached(cache=TTLCache(maxsize=1024, ttl=3600))
def get_ya_token(service_account_id, key_id, private_key):
    now = int(time())
    payload = {
        "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
        "iss": service_account_id,
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
    print(f"New IAM token, expires at {iam_token.json()['expiresAt']}")
    return iam_token.json()["iamToken"]


def check_questions(user_text):
        if len(user_text.split()) < 3:
            return False
        else:
            return True

        
def get_answer(user_text, answer, folder_id, service_account_id, key_id, private_key):
            
    values = {
        "modelUri": f"gpt://{folder_id}/yandexgpt-lite/latest",
        "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": "2000"},
        "messages": [
            {
                "role": "system",
                "text": "Тебе будет различные тексты и вопрос пользователя по ним. Тебе нужно ответить на вопрос по тексту. Если в тексте нет ответа, то напиши, что извините, я не знаю ответ на вопрос.  Так же, если в тексте мат, или подозрение на неприемлемый контент, ответь пользователю, что данный вопрос не является корректным. Под аббревиатурой фф или физфак имеется в виду физический факультет МГУ. Всегда будь вежлив и не используй мат. `  ",
            },
            {"role": "user", "text": "Ты работник колл центра, ответь на вопрос пользователя вежливо. Если не знаешь ответ или он некорректный, то напиши пользователю сообщение - 'Ващ запрос некоректный. 'Текст: " + answer + "Вопрос: " + user_text},
        ],
    }
    resp = r.post(
        f"https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        json=values,
        headers={
            "Authorization": f"Bearer {get_ya_token(service_account_id, key_id, private_key)}",
            "x-folder-id": folder_id,
        },
    )
    return resp.json()
