# Healthwork_ia
This is a local medical AI chatbot created in python using Llama 2, LangChain, Sentence Transformers, and Hugging Face.

**Make sure the machine you are running this on has atleast 16GB of ram**

Comunicates with a distant server for remote questions


Steps:
1. First download all the files. 
2. Open the folder in your desired code editor. 
3. Download this model https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin , then put in the folder. 
4. run cmd 
5. Activate virtual environment anaconda "conda activate env_name". 
6. Download all the dependencies from the "pip install -r requirements.txt" file. 
7. Now you are ready to run the Medical Chatbot hdx:
8. python ingest.py 
9. chainlit run model.py -w 
10. Actual usage for server :
11. chainlit run httpversion.py -w

