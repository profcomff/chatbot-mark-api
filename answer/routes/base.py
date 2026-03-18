import datetime
import logging
import sys


sys.path.append("../")

from auth_lib.fastapi import UnionAuth

from aiogram import Bot, Dispatcher
from aiogram.types import Update
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi_sqlalchemy import DBSessionMiddleware
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sqlalchemy import and_, desc
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from answer import __version__
from answer.bot.tg_bot.initialisation import bot_shutdown, bot_startup
from answer.models.db import Conversation, User
from answer.schemas.api_models import (
    ConversationContextResponse,
    CreateUserRequest,
    SaveConversationRequest,
    UserInput,
    UserResponse,
)
from answer.schemas.db_models import StatusMessage
from answer.services import get_search_service
from answer.settings import get_settings
from llm.llm import get_answer
from search.filter import length_filter
from search.nn import FilteredEnsembleRetriever, init_embedder
from search.preprocess import preprocess_stem, TextPreprocessor
from search.search import generate_keywords_dict, get_context, get_documents_from_qdrant


settings = get_settings()
search_service = get_search_service()
logger = logging.getLogger(__name__)

bot = None
dp = None

engine = create_engine(str(settings.DB_DSN), pool_pre_ping=True, pool_recycle=300)
Session = sessionmaker(bind=engine)

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


@app.post(settings.WEBHOOK_PATH)
async def webhook_handler(request: Request):
    """Обработчик webhook обновления от тг"""
    global bot, dp

    if not bot or not dp:
        logger.error("Bot or dispatcher not initialized")
        return {"status": "error", "message": "Bot not ready"}

    try:
        try:
            update_data = await request.json()
        except Exception as e:
            logger.error(f"Invalid JSON in webhook: {e}")
            return {"status": "error", "message": "Invalid JSON"}

        if not update_data:
            logger.error("Empty webhook data received")
            return {"status": "error", "message": "Empty data"}

        logger.info(f"Received webhook data with keys: {list(update_data.keys())}")

        try:
            update = Update(**update_data)
        except Exception as e:
            logger.error(f"Invalid Update object: {e}")
            return {"status": "error", "message": "Invalid update format"}

        logger.info(f"Created update object: {update}")

        await dp.feed_update(bot=bot, update=update)
        logger.info("Update processed successfully")

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.on_event("startup")
async def init_resources():
    global bot, dp

    bot, dp = await bot_startup()
    app.state.bot = bot

    app.state.embedder = init_embedder()

    app.state.qdrant_client = QdrantClient(
        url="http://qdrant.profcomff.com:6333",
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False
    )
    
    documents = get_documents_from_qdrant(
        client=app.state.qdrant_client,
        collection_name=settings.collection_name,
        page_content_field="page_content",
        metadata_field="metadata"
    )

    app.state.bm25_retriever = BM25Retriever.from_documents(
        documents, preprocess_func=preprocess_stem, k=settings.retrivier_k
    )

    app.state.vector_store = QdrantVectorStore(
    client=app.state.qdrant_client,  
    collection_name=settings.collection_name,
    embedding=app.state.embedder,
    )

    app.state.vector_retriever = app.state.vector_store.as_retriever(search_kwargs={"k": settings.retrivier_k})
    
    app.state.ensemble_retriever = EnsembleRetriever(retrievers=[app.state.bm25_retriever, app.state.vector_retriever], 
                                                    weights=[0.5, 0.5])
        
    app.state.filtered_ensemble_retriever = FilteredEnsembleRetriever(app.state.vector_store, 
                                                                      app.state.bm25_retriever, 
                                                                      retriever_k=settings.retrivier_k, 
                                                                      ensemble_k=settings.ensemble_k)
        
    app.state.keywords_dict = generate_keywords_dict(
        vector_store=app.state.vector_store, 
        output_json_path="file/key_words_dict.json"
    )
    
    app.state.text_preprocessor = TextPreprocessor.from_file()


    app_state_dict = {
        "embedder": app.state.embedder,
        "qdrant_client": app.state.qdrant_client,
        "vector_store": app.state.vector_store,
        "ensemble_retriever": app.state.ensemble_retriever,
        "keywords_dict": app.state.keywords_dict,
    }
    search_service.set_app_state(app_state_dict)


@app.on_event("shutdown")
async def shutdown_resources():
    global bot, dp
    await bot_shutdown()
    bot = None
    dp = None


@app.post("/greet")
async def generate_response(user_input: UserInput):
    if not user_input.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if user_input.generate_ai_response:
        ensemble_retriever = app.state.ensemble_retriever
    else:
        ensemble_retriever = app.state.filtered_ensemble_retriever
    
    processed_text = app.state.text_preprocessor.preprocess(user_input.text)
        
    results, combined_text = get_context(
        query=processed_text,
        key_words_dict=app.state.keywords_dict,
        ensemble_retriever=ensemble_retriever,
        vector_store=app.state.vector_store,
        ensemble_k=settings.ensemble_k,
        verbose=True,
    )
    
    formatted_results = [
        {
            "topic": getattr(r, 'topic', ''),
            "full_text": getattr(r, 'full_text', str(r)),
            "metadata": getattr(r, 'metadata', {})
        } 
        for r in results
    ]
    
    if user_input.generate_ai_response:
        if length_filter(text=user_input.text, max_len=settings.max_length):
            ai_answer = get_answer(
                context=combined_text, 
                question=user_input.text, 
                settings=settings,
            )
            
            response = {"results": formatted_results}
            if ai_answer:
                response["ai_answer"] = ai_answer
                
            return response
        else: 
            return {
                "results": [], 
                "ai_answer": 'Ваш запрос слишком длинный :( Сделайте короче или используйте режим без GPT.'
            }
    
    if len(formatted_results) > 0:
        return {"results": formatted_results}
    else:             
        return {
            "results": [], 
            "ai_answer": 'Извините, я не понял Ваш запрос. Попробуйте использовать GPT версию.'
        }       
    

@app.post("/users", response_model=UserResponse)
async def create_user(user_request: CreateUserRequest, user=Depends(UnionAuth())):
    """Создание нового пользователя"""
    try:
        with Session() as session:
            existing_user = session.query(User).filter(User.chat_id == user_request.chat_id).first()
            if existing_user:
                return UserResponse(
                    id=existing_user.id,
                    chat_id=existing_user.chat_id,
                    create_ts=existing_user.create_ts,
                    is_deleted=existing_user.is_deleted,
                )

            new_user = User(
                chat_id=user_request.chat_id, create_ts=datetime.datetime.now(datetime.timezone.utc), is_deleted=False
            )
            session.add(new_user)
            session.commit()
            session.refresh(new_user)

            logger.info(f"Создан новый пользователь с chat_id: {user_request.chat_id}")

            return UserResponse(
                id=new_user.id, chat_id=new_user.chat_id, create_ts=new_user.create_ts, is_deleted=new_user.is_deleted
            )

    except Exception as e:
        logger.error(f"Ошибка создания пользователя: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка создания пользователя")


@app.get("/users/{chat_id}", response_model=UserResponse)
async def get_user(chat_id: str, user=Depends(UnionAuth())):
    """Получение пользователя по chat_id"""
    try:
        with Session() as session:
            user = session.query(User).filter(User.chat_id == chat_id).one_or_none()
            if user is None:
                raise HTTPException(status_code=404, detail="Пользователь не найден")

            return UserResponse(id=user.id, chat_id=user.chat_id, create_ts=user.create_ts, is_deleted=user.is_deleted)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения пользователя: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка получения пользователя")


@app.get("/users/{chat_id}/context", response_model=ConversationContextResponse)
async def get_conversation_context(chat_id: str, user=Depends(UnionAuth())):
    """Получение контекста диалогов пользователя"""
    try:
        with Session() as session:
            user = session.query(User).filter(User.chat_id == chat_id).one_or_none()
            if user is None:
                raise HTTPException(status_code=404, detail="Пользователь не найден")

            conversations = (
                session.query(Conversation)
                .filter(and_(Conversation.user_id == user.id, Conversation.is_deleted == False))
                .order_by(desc(Conversation.create_ts))
                .limit(settings.CONTEXT_DEPTH)
                .all()
            )

            if not conversations:
                return ConversationContextResponse(context="", conversations_count=0)

            conversations = list(reversed(conversations))
            context_parts = []
            for conv in conversations:
                context_parts.append(f"Пользователь: {conv.request}")
                context_parts.append(f"Ассистент: {conv.response}")

            context_string = "\n".join(context_parts)

            return ConversationContextResponse(context=context_string, conversations_count=len(conversations))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения контекста диалогов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка получения контекста диалогов")


@app.post("/conversations")
async def save_conversation(conversation_request: SaveConversationRequest, user=Depends(UnionAuth())):
    """Сохранение диалога"""
    try:
        with Session() as session:
            user = session.query(User).filter(User.chat_id == conversation_request.user_chat_id).one_or_none()
            if user is None:
                raise HTTPException(status_code=404, detail="Пользователь не найден")

            conversation = Conversation(
                user_id=user.id,
                request=conversation_request.request,
                response=conversation_request.response,
                is_response_with_buttons=conversation_request.is_response_with_buttons,
                create_ts=datetime.datetime.now(datetime.timezone.utc),
                is_deleted=False,
            )

            session.add(conversation)
            session.commit()

            logger.info(f"Диалог сохранен для пользователя {conversation_request.user_chat_id}")

            return {"status": "success", "message": "Диалог успешно сохранен"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка сохранения диалога: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка сохранения диалога")


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
                                <div style="margin-top: 1em; padding-top: 0.5em; border-top: 1px solid #c8e6c9; font-style: italic; color: #666; font-size: 0.9em;">
                                    Ответ сгенерирован ИИ и может содержать неточности.
                                </div>
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
