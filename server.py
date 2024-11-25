from fastapi import FastAPI, Request
import uvicorn
import os

from utils import database_utils
from dotenv import load_dotenv

from utils.inference import inference as inference_fun
from utils.train import train as train_fun

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello, World!"}

@app.post("/train")
async def train(request: Request, username: str):
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')
    database_utils.load_str(tsv_str, username, skip_header=True)
    database_utils.save_tsv(tsv_str, "training/" + username)

    train_fun('keystroke_data.sqlite', username,
              rows_per_example=100, test_train_split=0, positive_negative_ratio=0)
    return {"message": "TSV data received successfully"}


@app.post("/inference")
async def inference(request: Request, username: str):
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')

    score, prediction = inference_fun(username, tsv_str)
    return {"message": "TSV data received successfully",
            "score": score,
            "prediction": prediction}


def main():
    load_dotenv()
    ssl_key_file = os.getenv("SSL_KEY_FILE_PATH")
    ssl_cert_file = os.getenv("SSL_CERT_FILE_PATH")
    print(ssl_key_file, ssl_cert_file)
    uvicorn.run(app, host="192.168.1.100", port=8000, ssl_keyfile=ssl_key_file, ssl_certfile=ssl_cert_file)


if __name__ == "__main__":
    # database_utils.drop_table()
    # database_utils.create_table()
    # database_utils.load_dir("datasets/training")
    main()

