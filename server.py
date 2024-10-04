import csv

from fastapi import FastAPI, Request
import uvicorn
from io import StringIO

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/upload-csv")
async def upload_csv(request: Request):
    # Read the CSV data from the request body
    csv_data = await request.body()

    # Decode the bytes to a string
    csv_str = csv_data.decode('utf-8')

    # Process the CSV data (for example, using Python's csv.reader)
    csv_reader = csv.reader(StringIO(csv_str))

    # Skip the header
    next(csv_reader)

    keypresses = []

    # Iterate through the CSV rows
    for row in csv_reader:
        keypress = {
            "key": row[0],
            "press_time": int(row[1]),
            "duration": int(row[2])
        }
        keypresses.append(keypress)

    # Do something with the data (e.g., save to a database)
    print(keypresses)

    return {"message": "CSV data received successfully"}


def main():
    uvicorn.run(app, host="192.168.1.100", port=8000)


if __name__ == "__main__":
    main()
