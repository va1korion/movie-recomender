import requests

print(requests.request('post', "http://127.0.0.1:8000/infer", json={
    'Date Rated': '2022-08-12',
    'Title': 'Earwig',
    'Title Type': 'movie',
    'IMDb Rating': 5.4,
    'Runtime (mins)': 114.0,
    'Genres': 'Drama, Fantasy, Horror',
    'Num Votes': '942',
    'Release Date': '2021-09-10',
    'Directors': 'Lucile Hadzihalilovic'
}).text)