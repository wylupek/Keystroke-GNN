import csv
from fastapi import FastAPI, Request
import uvicorn
from io import StringIO

from utils import database_utils
from utils.database_utils import print_tsv

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/upload-tsv")
async def upload_tsv(request: Request, username: str):
    # Read the TSV data from the request body
    tsv_data = await request.body()
    tsv_str = tsv_data.decode('utf-8')
    tsv_reader = csv.reader(StringIO(tsv_str), delimiter='\t')
    next(tsv_reader) # Skip headers

    key_presses = []
    for row in tsv_reader:
        if len(row) == 6:
            key_presses.append({
                "key": str(row[0]),
                "press_time": int(row[1]),
                "duration": int(row[2]),
                "accel_x": float(row[3]),
                "accel_y": float(row[4]),
                "accel_z": float(row[5]),
            })
    print(username)
    database_utils.print_tsv(key_presses)

    # Add the username from the query parameter
    database_utils.add_tsv_values(key_presses, username)

    return {"message": "TSV data received successfully"}


def main():
    uvicorn.run(app, host="192.168.1.100", port=8000)


if __name__ == "__main__":
    database_utils.drop_table()
    database_utils.setup_database()
    main()

