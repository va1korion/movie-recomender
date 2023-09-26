import os

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.responses import HTMLResponse, FileResponse, PlainTextResponse, RedirectResponse
import logging
import pickle

model = os.getenv("MODEL")
le= os.getenv("LABELER")
__version__ = "0.6.0"
app = FastAPI()
with open(model, 'rb') as f:
    regressor = pickle.load(f)

with open(le, 'rb') as f:
    labeler = pickle.load(f)


@app.get("/hello", response_class=PlainTextResponse)
async def read_hello(request: Request):
    context = {
        "request": request,
        "version": __version__,
    }
    return PlainTextResponse("Hello, I am a movie recommender. Please send your request as a JSON to "
                             "/infer to get a predicted score")


@app.get("/", response_class=RedirectResponse)
async def read_root(request: Request):
    context = {
        "request": request,
        "version": __version__,
    }
    return RedirectResponse("/index.html")


@app.get("/index.html", response_class=HTMLResponse)
async def read_index(request: Request):
    context = {
        "request": request,
        "version": __version__,
    }
    return HTMLResponse()


@app.post("/infer", response_class=PlainTextResponse)
async def infer(request: Request):
    context = {
        "request": request,
        "version": __version__,
    }
    return PlainTextResponse()


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
