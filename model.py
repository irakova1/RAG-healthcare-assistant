from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from chainlit.server import app
from fastapi import Request , HTTPException
from fastapi.responses import (
    HTMLResponse,
    JSONResponse
)
from fastapi import HTTPException, Request, Response
import json
from chainlit.context import init_http_context
import chainlit as cl
import socketio
import asyncio
import queue
#access_token = "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"

DB_FAISS_PATH = 'vectorstore/db_faiss'
#model_name_or_path = "TheBloke/llama-2-7b-chat.ggmlv3.q8_0.bin"
model_path = "llama-2-7b-chat.ggmlv3.q8_0.bin"
# model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = model_path,
        model_type="llama",
        max_new_tokens = 2048,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


def test(msg):
    print('callback !')
    print(msg)


# user_init = False
question_queue = queue.Queue()
question_counter = 0  # Initialize a counter for questions





# async def askquestion(data):
#     global question_counter
#     question_counter += 1
#     chain = cl.user_session.get("chainbot")
#     data['status'] = 'queued'
#     data['position'] = question_counter 
#     #await sio.emit('health_notif_status', data=data, namespace='/api/socket')
#     res = await chain.acall(data['question'])
#     data['res'] = res["result"]
#     question_counter -= 1
#     print(data , question_counter)
#     #await sio.emit('health_notif', data=data, namespace='/api/socket')

# @cl.on_chat_start
# def on_chat_start():
#     global user_init # Use the global flag variable
#     if not user_init:
#         user_init = True
#         chain = qa_bot()
#         cl.user_session.set("chainbot", chain)
#         print("User set for incoming questions")  
          
# @app.post("/endpoint")
# async def message( request: Request) :
#     try:
#         # Get the request body as bytes
#         request_body = await request.body()

#         # Parse the request body as JSON
#         request_json = json.loads(request_body)
#         print(request_json)
#         #except json.JSONDecodeError as e:
#         # raise HTTPException(status_code=400, detail="Invalid json request")
    
#     # Extracted message data
#         question = request_json.get("question" , None)
#         id = request_json.get("_id" , None)
#         chain = qa_bot()
#         await cl.user_session.set("chainbot", chain)
#         if question is None:
#             raise HTTPException(status_code=400 ,detail='Missing key for key_name')
    
#         # Create a response
#         response = { 
#             "message" : "Heres is the response" , 
#             "data" : request_json,
#             "type" : 'Success'
#         }
#         print(response)
#         asyncio.create_task(askquestion(request_json))
#         # Return the response
#         return response

#     except json.JSONDecodeError as e:
#         raise HTTPException(status_code=400, detail="Invalid json request")

@cl.on_chat_start
async def ask_question(data):
    global question_counter
    question_counter += 1
    chain = cl.user_session.get("chainbot")
    data['status'] = 'queued'
    data['position'] = question_counter 
    await sio.emit('health_notif_status', data=data, namespace='/api/socket')
    res = await chain.acall(data['question'])
    print(res['result'] , res)
    data['res'] = res["result"]
    data['status'] = 'resolved'
    question_counter -= 1
    await sio.emit('health_notif', data=data, namespace='/api/socket')

socket_initialized = False  # Add a global flag variable        
server_url = 'http://10.0.0.56:5000'  # Replace with your server's URL and ports 
sio = socketio.AsyncClient()

async def init():
    global socket_initialized  # Use the global flag variable
    if not socket_initialized:
        socket_initialized = True
        await sio.connect(server_url, namespaces=['/api/socket'])
        chain = qa_bot()
        cl.user_session.set("chainbot", chain)
        print("Starting socketio")

        @sio.event(namespace='/api/socket')
        async def connect():
            print("Connected to the server")

        @sio.event(namespace='/api/socket')
        async def message(data):
            print(f"Received message: {data}")

        @sio.event(namespace='/api/socket')
        async def health_chat(data):
            # Run the main function within an event loop
            await askquestion(data)

async def connect_to_server():
    try:
        await sio.connect(server_url, namespaces=['/api/socket'])
        print("Connected to the server")
    except Exception as e:
        print(f"Connection error: {e}")

@cl.on_chat_start
async def getdata():
    if not await connect_to_server():
        chain = qa_bot()
        cl.user_session.set("chainbot", chain)
        print("Session set for incoming questions")  

@sio.event(namespace='/api/socket')
async def message(data):
    print(f"Received message: {data}")

@sio.event(namespace='/api/socket')
async def health_chat(data):
    try:
        await ask_question(data)
    except Exception as e:
        print(f"Error in ask_question: {e}")
# asyncio.run(socket())

# async def socket():
#     print("Starting socketio")
#     sio = socketio.AsyncClient()
#     server_url = 'http://10.0.0.56:5000'  # Replace with your server's URL and ports
#     await sio.connect(server_url , namespaces=['/api/socket'])
#     @sio.event(namespace='/api/socket')
#     async def connect():
#         print("Connected to the server")

#     @sio.event(namespace='/api/socket')
#     async def message(data):
#         print(f"Received message: {data}")
    
#     @sio.event(namespace='/api/socket')
#     async def health_chat(data):
#         print(f"Received message: {data}")
#         print('DATA : ', data['question'])
#         await main(data['question'])

#     event = await sio.wait()



# @app.post("/api/ask")
# async def post_endpoint( request: Request) :
#     try:
#         request_body = await request.body()
#         request_json = json.loads(request_body)
#         question = request_json.get("subject" , None)
#         if question is None:
#             raise HTTPException(status_code=400 ,detail='Missing key for key_name')
#         #Maybe introduce a better system with a key on inital request must be returned or nothing happens on callback
      
#         async def callback(bot_res):
#             server_url = 'http://10.0.0.56:5000/api/health_history/callback'
#             headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
#             r = s.post(server_url, data=json.dumps(bot_res))
#             global processing_question
#             processing_question = False
#         connect = await login()
#         if connect is None:
#             raise HTTPException(status_code=400, detail="Server cannot connect to the ressource")
#         init_http_context()
#         # Create a response
#         value = cl.user_session.get("chainbot")
#         if value is None:
#             chain = qa_bot()
#             cl.user_session.set("chainbot", chain)
#             value = cl.user_session.get("chainbot")
#         request_json['status'] = 'queued'
#         global processing_question
#         if processing_question is True :
#             raise HTTPException(status_code=400, detail="A question is already being processed")
#         else:
#             processing_question = True
#             asyncio.create_task(ask_question(request_json , value , callback))
#             response = { 
#                 "message" : "Question has been queued for processing",
#                 "data" : request_json,
#                 "type" : 'Success'
#             }
#             # Return the response
#             return response
#     except json.JSONDecodeError as e:
#         raise HTTPException(status_code=400, detail="Invalid json request")
# Create a session with a cookie jar#     event = await sio.wait()
