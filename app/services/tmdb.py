import asyncio
import httpx
from typing import Dict, List, Optional

TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


class TMDBService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def _get(self, endpoint: str, params: Optional[dict] = None):
        if params is None:
            params = {}
        params["api_key"] = self.api_key

        response = await self.client.get(f"{TMDB_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()

    async def search_movie(self, movie_name: str) -> List[Dict]:
        data = await self._get("search/movie", {"query": movie_name, "language": "en-US"})
        results = data.get("results", [])[:3]

        return [
            {
                "id": m.get("id"),
                "title": m.get("title"),
                "year": m.get("release_date", "")[:4] if m.get("release_date") else "N/A",
            }
            for m in results
        ]

    async def fetch_reviews(self, movie_id: int, max_pages: int = 3) -> List[Dict]:
        tasks = [
            self._get(f"movie/{movie_id}/reviews", {"language": "en-US", "page": page})
            for page in range(1, max_pages + 1)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        reviews = []
        for response in responses:
            if isinstance(response, Exception):
                continue
            reviews.extend(response.get("results", []))

        return reviews

    async def fetch_movie_metadata(self, movie_id: int) -> Dict:
        movie_task = self._get(f"movie/{movie_id}", {"language": "en-US"})
        credits_task = self._get(f"movie/{movie_id}/credits")

        movie, credits = await asyncio.gather(movie_task, credits_task)

        genres = ", ".join(g["name"] for g in movie.get("genres", []))

        director = next(
            (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"),
            "Unknown",
        )

        cast = ", ".join(actor["name"] for actor in credits.get("cast", [])[:3])

        poster = POSTER_BASE_URL + movie["poster_path"] if movie.get("poster_path") else ""

        return {
            "title": movie.get("title"),
            "poster": poster,
            "rating": movie.get("vote_average"),
            "runtime": movie.get("runtime"),
            "genres": genres,
            "director": director,
            "overview": movie.get("overview"),
            "cast": cast,
        }

    async def get_movie_data(self, movie_id: int):
        metadata_task = self.fetch_movie_metadata(movie_id)
        reviews_task = self.fetch_reviews(movie_id)
        metadata, reviews = await asyncio.gather(metadata_task, reviews_task)
        return metadata, reviews

    async def close(self):
        await self.client.aclose()
