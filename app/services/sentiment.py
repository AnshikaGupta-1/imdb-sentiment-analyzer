import asyncio
import httpx
from typing import List, Dict


HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


class SentimentService:

    def __init__(self, hf_token: str):

        self.headers = {
            "Authorization": f"Bearer {hf_token}"
        }

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )

        # Limits how many HF requests run at once
        self.semaphore = asyncio.Semaphore(5)

    # ---------------------------------------------------------
    # Call HuggingFace API for a single piece of text
    # ---------------------------------------------------------

    async def _predict_one(self, text: str):

        payload = {
            "inputs": text
        }

        async with self.semaphore:

            response = await self.client.post(
                HF_API_URL,
                headers=self.headers,
                json=payload,
            )

            response.raise_for_status()

            return response.json()

    # ---------------------------------------------------------
    # Analyze Reviews
    # ---------------------------------------------------------

    async def analyze_reviews(
        self,
        reviews_raw: List[Dict],
    ):

        if not reviews_raw:

            return {
                "reviews": [],
                "positive_percent": 0,
                "negative_percent": 0,
                "positive_count": 0,
                "negative_count": 0,
            }

        texts = []

        usable_reviews = []

        for review in reviews_raw[:50]:

            content = review.get("content", "").strip()

            if not content:
                continue

            usable_reviews.append(review)

            texts.append(content[:300])

        if not texts:

            return {
                "reviews": [],
                "positive_percent": 0,
                "negative_percent": 0,
                "positive_count": 0,
                "negative_count": 0,
            }

        tasks = [
            self._predict_one(text)
            for text in texts
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        analyzed = []

        positive_count = 0

        for review, result in zip(usable_reviews, raw_results):

            if isinstance(result, Exception):
                print("HF prediction failed for one review:", result)
                continue

            # HF returns: [[{'label':'POSITIVE','score':...}, {'label':'NEGATIVE','score':...}]]
            prediction = result

            if isinstance(prediction, list) and len(prediction) > 0:
                prediction = prediction[0]

            if isinstance(prediction, list) and len(prediction) > 0:
                # top-scoring class first
                prediction = max(prediction, key=lambda p: p.get("score", 0))

            if not isinstance(prediction, dict) or "label" not in prediction:
                continue

            label = prediction["label"]

            sentiment = (
                "positive"
                if label == "POSITIVE"
                else "negative"
            )

            if sentiment == "positive":
                positive_count += 1

            analyzed.append(
                {
                    "content": review.get("content", ""),
                    "author": review.get(
                        "author",
                        "Anonymous",
                    ),
                    "rating": review.get(
                        "author_details",
                        {},
                    ).get(
                        "rating",
                        "N/A",
                    ),
                    "sentiment": label,
                    "label": sentiment,
                    "score": round(
                        prediction.get("score", 0),
                        3,
                    ),
                }
            )

        total = len(analyzed)

        if total == 0:

            positive_percent = 0

        else:

            positive_percent = round(
                (positive_count / total) * 100
            )

        negative_percent = 100 - positive_percent

        return {

            "reviews": analyzed,

            "positive_percent": positive_percent,

            "negative_percent": negative_percent,

            "positive_count": positive_count,

            "negative_count": total - positive_count,
        }

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    async def close(self):

        await self.client.aclose()
