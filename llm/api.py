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
                "text": "Тебе буден дан текст, ответь по нему пользователь задает вопрос, ты должен максимально точно ответить на него. Отвечай так, будто ты работник колл-центра и твоя работа отвечать на вопросы. Под аббревиатурой фф или физфак имеется в виду физический факультет МГУ.Если в тексте нет ответа, то напиши об этом.  Так же, если в тексте мат, то не отвечай.`  ",
            },
            {"role": "user", "text": "Текст: " + answer + " Вопрос: " + user_text},
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
