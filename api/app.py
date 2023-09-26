import os

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, FileResponse, PlainTextResponse, RedirectResponse
import logging
import pickle
import json


model = os.getenv("MODEL", default="../data/models/best_model.pkl")
le = os.getenv("LABELER", default="../data/preprocessing/label_encoder.pkl")
mm = os.getenv("MINMAX", default="../data/preprocessing/minmaxer.pkl")
__version__ = "0.6.5"


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

with open(model, 'rb') as f:
    regressor = pickle.load(f)

with open(le, 'rb') as f:
    labeler = pickle.load(f)

with open(mm, 'rb') as f:
    minmax = pickle.load(f)


@app.on_event('startup')
async def load_models():
    global regressor, labeler, minmax
    with open(model, 'rb') as f:
        regressor = pickle.load(f)

    with open(le, 'rb') as f:
        labeler = pickle.load(f)

    with open(mm, 'rb') as f:
        minmax = pickle.load(f)


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
    j = await request.json()
    df = pd.DataFrame([j])
    df.fillna(0)
    for column_name in df.columns:
        print(column_name)
        if column_name in ['Directors', 'Genres', 'Title Type', 'Title']:
            try:
                df[column_name] = labeler.transform(df[column_name])
            except ValueError:
                df[column_name] = pd.Series([0])
        if column_name in ['Date Rated', 'Release Date']:
            df[column_name] = pd.to_datetime(df[column_name], format='ISO8601')
            df[column_name] = minmax.transform(np.array(df[column_name]).reshape(1, -1))

    df['Unnamed: 0'] = pd.Series([0])
    names = regressor.feature_names_in_
    df = df[names]
    prediction = regressor.predict(df)
    context = {
        "request": request,
        "version": __version__,
    }
    return PlainTextResponse(str(prediction))


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
