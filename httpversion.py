import logging

from dotenv import load_dotenv

load_dotenv()

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import json
import asyncio
import requests
import os
import httpx

DB_FAISS_PATH = 'vectorstore/db_faiss'
model_path = "llama-2-7b-chat.ggmlv3.q8_0.bin"
s = requests.Session()
s.headers.update({'Content-type': 'application/json', 'Accept': 'text/plain', 'credentials': 'include'})
email = os.environ.get('EMAIL')
password = os.environ.get('PASSWORD')
server_url = os.environ.get('SERVER_URL')
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.INFO))


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=2048,
        temperature=0.5
    )
    return llm


# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


async def ask_question(data, bot, callback):
    res = await bot.acall(data['subject'])
    data['res'] = res["result"]
    data['status'] = 'completed'
    await callback(data)


session = httpx.AsyncClient()
headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'credentials': 'include'}


async def login():
    try:
        url = server_url + '/api/login'
        data = {"email": email, "password": password}
        response = await asyncio.wait_for(session.post(url, data=json.dumps(data), headers=headers), timeout=10)
        if response.status_code == 201:
            return True
        else:
            return None
    except asyncio.TimeoutError:
        print("Login request timed out. Server did not respond.")
        return None
    except httpx.RequestError as e:
        print(f"Login request failed: {e}")
        return None


async def get_last_question():
    url = server_url + '/api/health_history/last'
    response = await session.get(url, headers=headers)
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return None
    else:
        return None


async def patch_question_queue(data):
    url = server_url + '/api/health_history/status'
    response = await session.patch(url, data=json.dumps(data), headers=headers)
    print(response.status_code)
    if response.status_code == 201:
        try:
            return response.json()
        except json.JSONDecodeError as e:
            return None
    else:
        return None


async def callback(bot_res):
    url = server_url + '/api/health_history/callback'
    await session.put(url, data=json.dumps(bot_res), headers=headers)


async def main():
    logging.info('Starting server')
    chain = qa_bot()
    sleep_for = 0.5
    while True:
        sleep_for = min(sleep_for * 2, 10)
        try:
            connect = await login()
            if connect is None:
                logging.error('Server cannot connect to the resource')
            else:
                question = await get_last_question()
                if question is not None:
                    sleep_for = 0.5
                    logging.debug('new question')
                    queue_update = await patch_question_queue(question['data'])
                    if queue_update is None:
                        logging.error('Queue update failed')
                    else:
                        await ask_question(question['data'], chain, callback)
        except httpx.NetworkError as e:
            sleep_for = 10
            logging.exception(f'Network error: {e}')
        try:
            logging.debug(f'waiting for {sleep_for} seconds')
            await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
