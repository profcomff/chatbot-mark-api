import sys
import os

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
from nn.search import Bertinskii
import random


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


class UserInput(BaseModel):
    text: str

        
data = pd.read_excel(os.environ['ANSWER_DATA'])  
answer_embs = torch.load(os.environ['EMB_DATA'])

model_loading_status = "Модель загружается..."

try:
    model_loading_status = "Загрузка токенизатора и модели..."
    model_name = os.environ['EMB_MODEL']
    model = Bertinskii()
    model.load_tokenizer_model(model_name=model_name) 
    model_loading_status = "Модель успешно загружена!"
except Exception as e:
    model_loading_status = f"Ошибка загрузки модели: {str(e)}"
    
@app.post("/greet")
async def greet(user_input: UserInput):
    if not user_input.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    results = model.find_answer(user_question=user_input.text, answer_embs=answer_embs, answers_df=data)
    return {"results": results}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ассистент</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            :root {{
                --primary-color: #4A90E2;
                --secondary-color: #F5A623;
                --background: #f8f9fa;
                --text-color: #2d3436;
            }}
            
            * {{
                box-sizing: border-box;
                font-family: 'Roboto', sans-serif;
            }}
            
            body {{
                margin: 0;
                padding: 2rem;
                background: var(--background);
                color: var(--text-color);
            }}
            
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            
            h1 {{
                text-align: center;
                color: var(--primary-color);
                margin-bottom: 2rem;
                font-weight: 500;
            }}
            
            textarea {{
                width: 100%;
                height: 120px;
                padding: 1rem;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 16px;
                resize: vertical;
                transition: border-color 0.3s;
            }}
            
            textarea:focus {{
                outline: none;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
            }}
            
            .button-group {{
                display: flex;
                gap: 1rem;
                margin: 1rem 0;
            }}
            
            button {{
                padding: 0.8rem 1.5rem;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            #sendRequest {{
                background: var(--primary-color);
                color: white;
            }}
            
            #sendRequest:hover {{
                opacity: 0.9;
                transform: translateY(-1px);
            }}
            
            #clearInput {{
                background: #e0e0e0;
                color: var(--text-color);
            }}
            
            #clearInput:hover {{
                background: #d0d0d0;
            }}
            
            #modelStatus {{
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 8px;
                background: #e3f2fd;
                color: #1976d2;
                border-left: 4px solid #1976d2;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .topic {{
                padding: 1rem;
                margin: 1rem 0;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                cursor: pointer;
                transition: all 0.2s;
                border-left: 4px solid transparent;
            }}
            
            .topic:hover {{
                transform: translateX(5px);
                border-left-color: var(--primary-color);
            }}
            
            .full-text {{
                padding: 1rem;
                margin: 1rem 0;
                background: #f8f9fa;
                border-radius: 8px;
                white-space: pre-wrap;
                animation: fadeIn 0.3s ease;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(-10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .loader {{
                display: none;
                border: 4px solid #f3f3f3;
                border-top: 4px solid var(--primary-color);
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 1rem auto;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Марк</h1>
            <div id="modelStatus">📦 {model_loading_status}</div>
            
            <textarea 
                id="userInput" 
                placeholder="Введите ваш вопрос..."
                rows="3"
            ></textarea>
            
            <div class="button-group">
                <button id="sendRequest">
                    <svg style="width:20px;height:20px" viewBox="0 0 24 24">
                        <path fill="currentColor" d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
                    </svg>
                    Отправить
                </button>
                <button id="clearInput">
                    <svg style="width:20px;height:20px" viewBox="0 0 24 24">
                        <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
                    </svg>
                    Очистить
                </button>
            </div>
            
            <div class="loader" id="loader"></div>
            <div id="response"></div>
        </div>

        <script>
            function escapeHtml(unsafe) {{
                return unsafe
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }}

            function toggleFullText(index) {{
                const fullTextDiv = document.getElementById(`fullText${{index}}`);
                fullTextDiv.style.display = fullTextDiv.style.display === 'none' ? 'block' : 'none';
            }}

            async function handleSubmit() {{
                const userInput = document.getElementById('userInput').value;
                const responseDiv = document.getElementById('response');
                const loader = document.getElementById('loader');
                
                responseDiv.innerHTML = '';
                loader.style.display = 'block';
                
                try {{
                    const response = await fetch('/greet', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ text: userInput }})
                    }});

                    const data = await response.json();
                    loader.style.display = 'none';

                    if (data.results) {{
                        data.results.forEach((result, index) => {{
                            const topicDiv = document.createElement('div');
                            topicDiv.className = 'topic';
                            topicDiv.innerHTML = `
                                <strong>${{escapeHtml(result.topic)}}</strong>
                                <div style="color:#666; font-size:0.9em; margin-top:0.5em">Нажмите для просмотра полного текста</div>
                            `;
                            topicDiv.onclick = () => toggleFullText(index);
                            
                            const fullTextDiv = document.createElement('div');
                            fullTextDiv.id = `fullText${{index}}`;
                            fullTextDiv.className = 'full-text';
                            fullTextDiv.style.display = 'none';
                            fullTextDiv.innerHTML = `
                                <div style="color: var(--primary-color); margin-bottom: 0.5rem;">🔍 Полный текст:</div>
                                <div>${{escapeHtml(result.full_text)}}</div>
                            `;
                            
                            responseDiv.appendChild(topicDiv);
                            responseDiv.appendChild(fullTextDiv);
                        }});
                    }} else if (data.message) {{
                        responseDiv.textContent = data.message;
                    }}
                }} catch (error) {{
                    loader.style.display = 'none';
                    responseDiv.textContent = 'Ошибка: ' + error.message;
                }}
            }}

            // Обработчики событий
            document.getElementById('sendRequest').addEventListener('click', handleSubmit);
            
            document.getElementById('clearInput').addEventListener('click', () => {{
                document.getElementById('userInput').value = '';
                document.getElementById('response').innerHTML = '';
            }});

            // Обработка Enter
            document.getElementById('userInput').addEventListener('keypress', (e) => {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    handleSubmit();
                }}
            }});
        </script>
    </body>
    </html>
    """