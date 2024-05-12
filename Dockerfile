FROM python:3.11

WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN curl "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin?download=true" -L -o llama-2-7b-chat.ggmlv3.q8_0.bin
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY vectorstore vectorstore
COPY .chainlit .
COPY chainlit.md .
COPY httpversion.py .

EXPOSE 8080

CMD ["python", "httpversion.py"]
