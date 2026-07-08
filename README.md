# MovieSentiment 🎬

[![Live Demo](https://img.shields.io/badge/Live%20Demo-View%20App-f3d03e?style=for-the-badge)](https://imdb-sentiment-analyzer-beta.vercel.app/)

**Search any movie, pull its real TMDb reviews, and see what audiences actually think — broken down by sentiment, in seconds.**

A FastAPI web app that fetches audience reviews for any movie from TMDb and classifies each one using a RoBERTa transformer served via the Hugging Face Inference API, then surfaces a clean visual breakdown alongside the movie's details.

<img width="1897" height="910" alt="image" src="https://github.com/user-attachments/assets/634b9f84-6f14-4c9a-ae96-a3af81877f99" />

## ✨ Features

- 🔍 **Live movie search** powered by the TMDb API, with instant title matching
- 🤖 **AI sentiment classification** on real reviews using `cardiffnlp/twitter-roberta-base-sentiment-latest` (positive / neutral / negative), via the Hugging Face Inference API
- 📊 **Confidence-weighted overall score** — a single 0–100 number combining every review's label *and* the model's certainty, not just a raw positive/negative count
- 🎞️ Rich movie metadata: poster, rating, runtime, genre, director, and cast
- ⚡ **Async, concurrent review processing** with `httpx` + `asyncio` for fast response times
- 🗃️ **In-memory TTL caching** to avoid redundant TMDb/HF calls on repeat lookups
- 📱 Responsive UI, no frontend framework required

## 🖼️ Demo

| Movie details + overall score | Sentiment breakdown + reviews |
|---|---|
| <img width="1897" height="907" alt="image" src="https://github.com/user-attachments/assets/bff30a62-6717-4c44-bceb-f82947e959c9" /> | <img width="1901" height="906" alt="image" src="https://github.com/user-attachments/assets/f041fe97-ddad-4b9a-9a1b-82cc8d035042" /> |

## 🛠️ Tech Stack

**Backend:** FastAPI · Python `asyncio` · Jinja2 templating
**AI/ML:** Hugging Face Inference API · `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa)
**Data:** TMDb API (search, reviews, credits)
**Infra:** `httpx` async client, in-memory TTL cache, deployed on Vercel

## ⚙️ How It Works

1. User searches a movie title → TMDb search API returns matching titles
2. On selecting a movie, the app concurrently fetches metadata (poster, cast, genre, etc.) and up to 3 pages of reviews
3. Each review is sent to the Hugging Face Inference API for classification, with retry logic for cold-start (`503`) responses
4. Results are aggregated into positive/neutral/negative percentages and a confidence-weighted overall score
5. Everything is cached for an hour so repeat lookups are instant

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/imdb-sentiment-analyzer.git
cd imdb-sentiment-analyzer
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with:

```
TMDB_API_KEY=your_tmdb_api_key
HF_API_TOKEN=your_huggingface_token
```

Run locally:

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000`.

## 📁 Project Structure

```
├── app/
│   ├── main.py                # FastAPI routes
│   ├── services/
│   │   ├── tmdb.py            # TMDb API client
│   │   ├── sentiment.py       # HF sentiment classification
│   │   └── cache.py           # In-memory TTL cache
│   └── templates/
│       └── index.html
├── requirements.txt
└── vercel.json
```

## 👤 Author

**Anshika Gupta**

---

⭐ If you found this project interesting, consider giving it a star!

