# The following is the code of fastAPI which containes a simple
# get api to read the log file of the model

from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
import os,uvicorn

app=FastAPI()

file_path="/opt/SpeechEmotionRecognition/utils/logs.txt"

@app.get("/log_status")
async def fetch_logs():
    """
    The following api is used to get the:
    1) Status of the logs
    2) Pass the contents of the logs 
    """
    if os.path.exists(file_path):
        with open(file_path,'r') as log_file:
            logs=log_file.read()
        return JSONResponse(content={"logs":logs})
    else:
        raise HTTPException(status_code=404,detail="Log file not found")

if "__main__" == __name__:
    uvicorn.run(app,host="3.94.211.50",port=8000)