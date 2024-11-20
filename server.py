from fastapi import FastAPI, Request
import uvicorn
import os

from utils import database_utils
from dotenv import load_dotenv

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/upload-tsv")
async def upload_tsv(request: Request, username: str):
    # Read the TSV data from the request body
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')
    database_utils.load_str(tsv_str, username, skip_header=True)
    return {"message": "TSV data received successfully"}


def main():
    load_dotenv()
    ssl_key_file = os.getenv("SSL_KEY_FILE_PATH")
    ssl_cert_file = os.getenv("SSL_CERT_FILE_PATH")
    print(ssl_key_file, ssl_cert_file)
    uvicorn.run(app, host="192.168.1.100", port=8000, ssl_keyfile=ssl_key_file, ssl_certfile=ssl_cert_file)


if __name__ == "__main__":
    database_utils.drop_table()
    database_utils.create_table()
    main()

