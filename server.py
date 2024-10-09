import csv
from fastapi import FastAPI, Request
import uvicorn
from io import StringIO

import utils

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/upload-csv")
async def upload_csv(request: Request):
    # Read the CSV data from the request body
    csv_data = await request.body()
    csv_str = csv_data.decode('utf-8')
    csv_reader = csv.reader(StringIO(csv_str))
    # Skip the header
    next(csv_reader)

    key_presses = []
    for row in csv_reader:
        key_presses.append({
            "key": row[0],
            "press_time": int(row[1]),
            "duration": int(row[2])
        })

    # TODO Add username from login
    utils.add_csv_values(key_presses, "USER")

    return {"message": "CSV data received successfully"}


def main():
    uvicorn.run(app, host="192.168.1.100", port=8000)


if __name__ == "__main__":
    utils.setup_database()
    main()
    utils.drop_table()
