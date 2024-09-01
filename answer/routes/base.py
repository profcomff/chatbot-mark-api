import sys

sys.path.append("../")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi_sqlalchemy import DBSessionMiddleware
from answer import __version__
from answer.settings import get_settings

import pandas as pd
import json
import torch
from nn.init import load_model
from nn.search import make_query_emb, get_similar_context
from llm.api import get_answer, check_questions


settings = get_settings()
app = FastAPI(
    title='Ассистент',
    description='-',
    version=__version__,

    root_path=settings.ROOT_PATH if __version__ != 'dev' else '',
    docs_url=None if __version__ != 'dev' else '/docs',
    redoc_url=None,
)

app.add_middleware(
    DBSessionMiddleware,
    db_url=str(settings.DB_DSN),
    engine_args={"pool_pre_ping": True, "isolation_level": "AUTOCOMMIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# Модель для запроса
class UserInput(BaseModel):
    text: str

        
data = pd.read_csv("/ff/file/context.csv")["QA"]
    
with open("/ff/file/key.json") as f:
    settings = json.load(f)
service_account_id = settings["service_account_id"]
key_id = settings["id"]
private_key = settings["private_key"]
folder_id = "b1ggivrnbg1ftsr8no1s"
    
tokenizer, model = load_model()
    
context_embs = torch.load("/ff/file/context.pt")

@app.post("/greet")
async def greet(user_input: UserInput):
    if not user_input.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    query_emb = make_query_emb(user_input.text, model, tokenizer)
    similar_context = get_similar_context(data, query_emb, context_embs)

    check = check_questions(user_input.text)
    if check:
        answer = get_answer(
            user_input.text, similar_context, folder_id, service_account_id, key_id, private_key
        )["result"]["alternatives"][0]["message"]["text"]
        return {"message": answer}

        
    else:
        return {"message": 'Привет! Уточни свой вопрос.'}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ассистент</title>
    </head>
    <body>
        <h1>Ассистент</h1>
        <input type="text" id="userInput" placeholder="Введите текст">
        <button id="sendRequest">Отправить</button>
        
        <div id="response"></div>

        <script>
            document.getElementById('sendRequest').addEventListener('click', async () => {
                const userInput = document.getElementById('userInput').value;

                // Отправка POST-запроса на сервер
                try {
                    const response = await fetch('/greet', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: userInput })
                    });

                    if (!response.ok) {
                        throw new Error('Network response was not ok ' + response.statusText);
                    }

                    const data = await response.json();
                    document.getElementById('response').innerText = data.message;

                } catch (error) {
                    document.getElementById('response').innerText = 'Ошибка: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    """