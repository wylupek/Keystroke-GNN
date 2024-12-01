from fastapi import FastAPI, Request
import uvicorn
from utils import database_utils
from utils.inference import inference as inference_fun
from utils.train import train as train_fun, LoadMode

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello, World!"}


@app.post("/upload_tsv")
async def upload_tsv(request: Request, username: str):
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')

    if not database_utils.load_str(tsv_str, username, skip_header=True):
        return {"message": "Error: Couldn't load the data"}
    database_utils.save_tsv(content=tsv_str, base_path="./datasets/training/", username=username)

    return {"message": "TSV data received successfully"}


@app.post("/train")
async def train(request: Request, username: str):
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')

    if not database_utils.load_str(tsv_str, username, skip_header=True):
        return {"message": "Error: Couldn't load the data"}
    database_utils.save_tsv(content=tsv_str, base_path="./datasets/training/", username=username)

    train_fun('keystroke_data.sqlite', username, mode=LoadMode.ONE_HOT,
              test_train_split=0, positive_negative_ratio=1, hidden_dim=128,
              rows_per_example=50, offset=10)

    return {"message": "TSV data received successfully. Training succeeded."}


@app.post("/inference")
async def inference(request: Request, username: str):
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')

    score, prediction = inference_fun(username, tsv_str)
    return {"message": "TSV data received successfully. Inference succeeded",
            "score": score,
            "prediction": prediction}


def main():
    ssl_key_file = "ssl/key.pem"
    ssl_cert_file = "ssl/cert.pem"
    print(ssl_key_file, ssl_cert_file)
    uvicorn.run(app, host="192.168.1.100", port=8000, ssl_keyfile=ssl_key_file, ssl_certfile=ssl_cert_file)


if __name__ == "__main__":
    database_utils.drop_table()
    database_utils.create_table()
    database_utils.load_dir("datasets/training")
    main()

