import asyncio
import httpx
from typing import List, Dict


# cardiffnlp/twitter-roberta-base-sentiment-latest is reliably kept warm on the
# hf-inference provider and returns 3 labels: "negative", "neutral", "positive".
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


class SentimentService:

    def __init__(self, hf_token: str):

        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "x-wait-for-model": "true",
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
    # (with one retry on transient failures)
    # ---------------------------------------------------------

    async def _predict_one(self, text: str, review_index: int = -1):

        payload = {"inputs": text}

        async with self.semaphore:

            for attempt in range(2):

                try:
                    response = await self.client.post(
                        HF_API_URL,
                        headers=self.headers,
                        json=payload,
                    )

                    if response.status_code == 503:
                        print(f"[review {review_index}] 503 model loading, attempt {attempt + 1}")
                        await asyncio.sleep(3)
                        continue

                    if response.status_code != 200:
                        print(
                            f"[review {review_index}] HF error "
                            f"{response.status_code}: {response.text[:300]}"
                        )
                        response.raise_for_status()

                    return response.json()

                except httpx.HTTPStatusError:
                    raise
                except Exception as e:
                    print(f"[review {review_index}] request failed: {e}")
                    if attempt == 1:
                        raise
                    await asyncio.sleep(2)

            raise RuntimeError(f"[review {review_index}] model did not load in time")

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
                "neutral_percent": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "overall_score": 50,
                "overall_label": "No Data",
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
                "neutral_percent": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "overall_score": 50,
                "overall_label": "No Data",
            }

        tasks = [
            self._predict_one(text, review_index=i)
            for i, text in enumerate(texts)
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        analyzed = []

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        # Running sum of signed, confidence-weighted scores, used to derive
        # a single overall_score below (+score for positive, -score for
        # negative, 0 for neutral).
        signed_score_sum = 0.0

        for review, result in zip(usable_reviews, raw_results):

            if isinstance(result, Exception):
                print("Skipping review, prediction failed:", result)
                continue

            # HF returns either:
            # [{'label':'positive','score':...}, {'label':'neutral','score':...}, {'label':'negative','score':...}]
            # or nested: [[{'label':'positive','score':...}, ...]]
            prediction = result

            if isinstance(prediction, list) and len(prediction) > 0:
                if isinstance(prediction[0], list):
                    prediction = prediction[0]

            if isinstance(prediction, list) and len(prediction) > 0:
                prediction = max(prediction, key=lambda p: p.get("score", 0))

            if not isinstance(prediction, dict) or "label" not in prediction:
                print("Unexpected HF response shape, skipping:", result)
                continue

            label = prediction["label"]

            # cardiffnlp/twitter-roberta-base-sentiment-latest returns lowercase
            # "positive" / "neutral" / "negative" directly. Some deployments may
            # instead surface generic "LABEL_0/1/2" ids, so we handle both.
            label_map = {
                "positive": "positive",
                "neutral": "neutral",
                "negative": "negative",
                "label_2": "positive",
                "label_1": "neutral",
                "label_0": "negative",
            }

            sentiment = label_map.get(label.lower(), "neutral")

            confidence = prediction.get("score", 0)

            if sentiment == "positive":
                positive_count += 1
                signed_score_sum += confidence
            elif sentiment == "negative":
                negative_count += 1
                signed_score_sum -= confidence
            else:
                neutral_count += 1
                # neutral contributes 0, it neither pulls the score up nor down

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
            negative_percent = 0
            neutral_percent = 0
            overall_score = 50
            overall_label = "No Data"

        else:

            positive_percent = round((positive_count / total) * 100)
            negative_percent = round((negative_count / total) * 100)
            # avoid rounding drift so the three percentages sum to 100
            neutral_percent = 100 - positive_percent - negative_percent

            # signed_score_sum / total sits in roughly [-1, 1]; rescale to
            # [0, 100] so 50 reads as neutral, 100 as unanimously confident
            # positive, and 0 as unanimously confident negative.
            average_signed_score = signed_score_sum / total
            overall_score = round(((average_signed_score + 1) / 2) * 100)
            overall_score = max(0, min(100, overall_score))

            if overall_score >= 75:
                overall_label = "Very Positive"
            elif overall_score >= 60:
                overall_label = "Positive"
            elif overall_score >= 40:
                overall_label = "Mixed"
            elif overall_score >= 25:
                overall_label = "Negative"
            else:
                overall_label = "Very Negative"

        return {

            "reviews": analyzed,

            "positive_percent": positive_percent,

            "negative_percent": negative_percent,

            "neutral_percent": neutral_percent,

            "positive_count": positive_count,

            "negative_count": negative_count,

            "neutral_count": neutral_count,

            "overall_score": overall_score,

            "overall_label": overall_label,
        }

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    async def close(self):

        await self.client.aclose()
