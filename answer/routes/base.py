import sys
import os
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

from transformers import XLMRobertaTokenizer, XLMRobertaModel
from langchain_chroma import Chroma
from nltk.stem.snowball import SnowballStemmer
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from llm.llm import get_answer
import json
import torch
from nn.search import get_context, E5LangChainEmbedder, preprocess
import random
import pickle


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
    generate_ai_response: bool = False  

        
@app.on_event("startup")
def init_resources():    
    with open(settings.GIGA_KEY_PATH, "r") as f:
        app.state.credentials = f.read().strip()
    
    app.state.tokenizer = XLMRobertaTokenizer.from_pretrained("d0rj/e5-base-en-ru", use_cache=False)
    app.state.model = XLMRobertaModel.from_pretrained("d0rj/e5-base-en-ru", use_cache=False)
    
    embedder = E5LangChainEmbedder(
        tokenizer=app.state.tokenizer,
        model=app.state.model,
    )
    
    app.state.vector_store = Chroma(
        collection_name="docs",
        embedding_function=embedder,
        persist_directory=settings.CHROMA_DIR
    )
    
    all_docs = app.state.vector_store.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=doc_text, metadata=metadata)
        for doc_text, metadata in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    
    app.state.bm25_retriever = BM25Retriever.from_documents(
        documents, 
        preprocess_func=preprocess
    )
        
        
@app.post("/greet")
async def generate_response(user_input: UserInput):
    if not user_input.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    results, combined_text = get_context(
        query=user_input.text,
        tokenizer=app.state.tokenizer,
        model=app.state.model,
        bm_25=app.state.bm25_retriever,
        vector_store=app.state.vector_store,
        ensemble_k=settings.ensemble_k,
        retrivier_k=settings.retrivier_k
    )
    
    if user_input.generate_ai_response:
        ai_answer = get_answer(
            context=combined_text, 
            question=user_input.text, 
            credentials=app.state.credentials,
            settings=settings
        )
        return {"results": results, "ai_answer": ai_answer}
    
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

            #aiResponse {{
                background: #4CAF50;
                color: white;
            }}

            #aiResponse:hover {{
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

            .ai-answer {{
                padding: 1rem;
                margin: 1rem 0;
                background: #e8f5e9;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
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
            
            /* Добавленные стили для сообщений */
            .warning-message {{
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 8px;
                background: #fff3e0;
                color: #e65100;
                border-left: 4px solid #f57c00;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .info-message {{
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Марк</h1>            
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
                    Поиск документов
                </button>
                <button id="aiResponse">
                    <svg style="width:20px;height:20px" viewBox="0 0 24 24">
                        <path fill="currentColor" d="M18,16H6V4H18M18,2H6A2,2 0 0,0 4,4V16A2,2 0 0,0 6,18H18A2,2 0 0,0 20,16V4A2,2 0 0,0 18,2M22,6V20H24V6H22M11,12H13V14H11V12M11,8H13V10H11V8M11,16H13V18H11V16Z" />
                    </svg>
                    Ответ AI
                </button>
                <button id="clearInput">
                    <svg style="width:20px;height:20px" viewBox="0 0 24 24">
                        <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
                    </svg>
                    Очистить
                </button>
            </div>
            
            <div class="loader" id="loader"></div>
            <div id="response">
                <div class="info-message">
                    <svg style="width:24px;height:24px" viewBox="0 0 24 24">
                        <path fill="currentColor" d="M11,9H13V7H11M12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M11,17H13V11H11V17Z" />
                    </svg>
                    <div>Задайте вопрос в поле выше, и я постараюсь найти ответ</div>
                </div>
            </div>
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

            async function handleSubmit(generateAI = false) {{
                const userInput = document.getElementById('userInput').value;
                const responseDiv = document.getElementById('response');
                const loader = document.getElementById('loader');
                
                // Проверка пустого ввода
                if (!userInput.trim()) {{
                    responseDiv.innerHTML = `
                        <div class="warning-message">
                            <svg style="width:24px;height:24px" viewBox="0 0 24 24">
                                <path fill="currentColor" d="M11 15H13V17H11V15M11 7H13V13H11V7M12 2C6.47 2 2 6.5 2 12A10 10 0 0 0 12 22A10 10 0 0 0 22 12A10 10 0 0 0 12 2M12 20A8 8 0 0 1 4 12A8 8 0 0 1 12 4A8 8 0 0 1 20 12A8 8 0 0 1 12 20Z" />
                            </svg>
                            <div>Пожалуйста, введите ваш вопрос</div>
                        </div>
                    `;
                    return;
                }}
                
                responseDiv.innerHTML = '';
                loader.style.display = 'block';
                
                try {{
                    const response = await fetch('/greet', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ 
                            text: userInput,
                            generate_ai_response: generateAI 
                        }})
                    }});

                    const data = await response.json();
                    loader.style.display = 'none';

                    // Обработка случая, когда максимальный score < 0.8
                    if (data.message) {{
                        responseDiv.innerHTML = `
                            <div class="warning-message">
                                <svg style="width:24px;height:24px" viewBox="0 0 24 24">
                                    <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                                </svg>
                                <div>${{escapeHtml(data.message)}}</div>
                            </div>
                        `;
                    }} 
                    // Обработка результатов
                    else if (data.results) {{
                        // Если есть AI ответ - показываем его первым
                        if (data.ai_answer) {{
                            const aiDiv = document.createElement('div');
                            aiDiv.className = 'ai-answer';
                            aiDiv.innerHTML = `
                                <div style="color: #2E7D32; margin-bottom: 0.5rem;">🤖 Ответ AI:</div>
                                <div>${{escapeHtml(data.ai_answer)}}</div>
                            `;
                            responseDiv.appendChild(aiDiv);
                        }}

                        // Показываем результаты поиска
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
                    }}
                }} catch (error) {{
                    loader.style.display = 'none';
                    responseDiv.innerHTML = `
                        <div class="warning-message">
                            <svg style="width:24px;height:24px" viewBox="0 0 24 24">
                                <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                            </svg>
                            <div>Произошла ошибка: ${{escapeHtml(error.message)}}</div>
                        </div>
                    `;
                }}
            }}

            // Обработчики событий
            document.getElementById('sendRequest').addEventListener('click', () => handleSubmit(false));
            document.getElementById('aiResponse').addEventListener('click', () => handleSubmit(true));
            
            document.getElementById('clearInput').addEventListener('click', () => {{
                document.getElementById('userInput').value = '';
                document.getElementById('response').innerHTML = `
                    <div class="info-message">
                        <svg style="width:24px;height:24px" viewBox="0 0 24 24">
                            <path fill="currentColor" d="M11,9H13V7H11M12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M11,17H13V11H11V17Z" />
                        </svg>
                        <div>Задайте вопрос в поле выше, и я постараюсь найти ответ</div>
                    </div>
                `;
            }});

            // Обработка Enter
            document.getElementById('userInput').addEventListener('keypress', (e) => {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    handleSubmit(false);
                }}
            }});
        </script>
    </body>
    </html>
    """