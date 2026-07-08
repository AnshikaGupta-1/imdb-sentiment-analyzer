import os
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.tmdb import TMDBService
from app.services.sentiment import SentimentService
from app.services.cache import MemoryCache

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

tmdb_service = TMDBService(TMDB_API_KEY) if TMDB_API_KEY else None
sentiment_service = SentimentService(HF_API_TOKEN) if HF_API_TOKEN else None
cache = MemoryCache(ttl_seconds=3600)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("MovieSentiment started.")
    yield
    if tmdb_service:
        await tmdb_service.close()
    if sentiment_service:
        await sentiment_service.close()
    print("MovieSentiment shutdown complete.")


app = FastAPI(title="MovieSentiment", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _missing_keys_error():
    missing = []
    if not TMDB_API_KEY:
        missing.append("TMDB_API_KEY")
    if not HF_API_TOKEN:
        missing.append("HF_API_TOKEN")
    return f"Missing environment variable(s): {', '.join(missing)}. Set them in Vercel Project Settings > Environment Variables."


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    error = _missing_keys_error() if (not tmdb_service or not sentiment_service) else None
    return templates.TemplateResponse("index.html", {"request": request, "error": error})


@app.post("/search", response_class=HTMLResponse)
async def search_movie(request: Request, movie: str = Form(...)):
    if not tmdb_service:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": _missing_keys_error()}
        )

    movie = movie.strip()

    if not movie:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": "Please enter a movie name."}
        )

    try:
        movies = await tmdb_service.search_movie(movie)

        if not movies:
            return templates.TemplateResponse(
                "index.html", {"request": request, "error": "No matching movies found."}
            )

        return templates.TemplateResponse(
            "index.html", {"request": request, "movies": movies}
        )

    except Exception as e:
        print("Search Error:", e)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Unable to search movies at the moment."},
        )


@app.get("/movie/{movie_id}", response_class=HTMLResponse)
async def movie_details(request: Request, movie_id: int):
    if not tmdb_service or not sentiment_service:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": _missing_keys_error()}
        )

    cache_key = f"movie_{movie_id}"
    cached_result = cache.get(cache_key)

    if cached_result is not None:
        print("Serving from cache.")
        cached_result = {**cached_result, "request": request}
        return templates.TemplateResponse("index.html", cached_result)

    try:
        metadata, reviews_raw = await tmdb_service.get_movie_data(movie_id)

        if not reviews_raw:
            result = {
                "request": request,
                **metadata,
                "reviews": [],
                "positive_percent": 0,
                "negative_percent": 0,
                "positive_count": 0,
                "negative_count": 0,
                "error": "No reviews available for this movie.",
            }
            cache.set(cache_key, {k: v for k, v in result.items() if k != "request"})
            return templates.TemplateResponse("index.html", result)

        sentiment_result = await sentiment_service.analyze_reviews(reviews_raw)

        result = {
            "request": request,
            **metadata,
            **sentiment_result,
        }

        cache_result = {k: v for k, v in result.items() if k != "request"}
        cache.set(cache_key, cache_result)

        return templates.TemplateResponse("index.html", result)

    except Exception as e:
        print("Movie Details Error:", e)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Failed to fetch movie details or perform sentiment analysis. Please try again.",
            },
        )
