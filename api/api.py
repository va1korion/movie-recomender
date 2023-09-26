import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, FileResponse, PlainTextResponse, RedirectResponse
import logging
import pickle

model = os.getenv("MODEL")
le = os.getenv("LABELER")
__version__ = "0.6.0"
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

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
    return templates.TemplateResponse("index.html", context)


@app.post("/infer", response_class=PlainTextResponse)
async def infer(request: Request):
    json = await request.json()
    data = pd.read_json(json)
    labeled = labeler.predict(data)
    prediction = regressor.predict(labeled)
    context = {
        "request": request,
        "version": __version__,
    }
    return PlainTextResponse(prediction)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
