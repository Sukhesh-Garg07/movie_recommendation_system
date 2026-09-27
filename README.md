# Movie Recommendation System

A hybrid movie recommender built on the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) dataset, combining item-based collaborative filtering with content-based filtering (genres + user tags), served through a FastAPI endpoint.

## How it works

1. **Collaborative filtering (CF)** — builds a user-item ratings matrix, mean-centers each user's ratings (removes bias from generous/harsh raters), and computes item-item cosine similarity. A user's score for an unseen movie is a similarity-weighted sum over movies they've already rated.
2. **Content-based filtering** — combines each movie's genres and all user-submitted tags into a text blob, vectorizes it with TF-IDF, and builds a "taste profile" per user from their positively-rated movies. Scores every movie by cosine similarity to that profile.
3. **Hybrid blend** — final score = `ALPHA * CF_score + (1 - ALPHA) * content_score` (both min-max normalized first). Default `ALPHA = 0.7`. The content signal keeps recommendations meaningful even when a user's CF signal is sparse.
4. **Cold-start fallback** — for a `user_id` not in the dataset, falls back to the highest-average-rated movies with at least 20 ratings (avoids one 5★ rating looking like "the best movie ever").

## Project structure

```
movie_rec_system/
├── data/                  # or flat, depending on DATA_DIR in data_loader.py
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv          # loaded but not currently used in scoring
├── data_loader.ipynb          # loads CSVs, builds combined genre+tag content text per movie
├── recommender.ipynb          # MovieRecommender class: CF + content hybrid scoring
├── api.py                  # FastAPI wrapper exposing /recommend/{user_id}
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv` either flat next to the scripts or in a `data/` folder — matching whatever `DATA_DIR` is set to in `data_loader.py`.

## Usage

**Interactive CLI (for testing/exploring):**
```bash
python recommender.py
```
Prompts for a user ID and a number of recommendations, prints titles.

**API server:**
```bash
uvicorn api:app --reload
```
- `GET /recommend/{user_id}?top_n=10` — returns JSON recommendations
- `GET /docs` — interactive Swagger UI for testing without curl/Postman

Example:
```bash
curl "http://127.0.0.1:8000/recommend/1?top_n=5"
```
```json
{
  "user_id": 1,
  "recommendations": [
    {"movie_id": 7248, "title": "Suriyothai (a.k.a. Legend of Suriyothai, The) (2001)"},
    {"movie_id": 4103, "title": "Empire of the Sun (1987)"}
  ]
}
```

## Dataset

MovieLens ml-latest-small: 100,836 ratings, 3,683 tag applications, 9,742 movies, 610 users (March 1996 – Sept 2018). Note: ~18 movies exist in `movies.csv` with no ratings at all, so they never appear in `movie_index` and can't currently be recommended.

## Known limitations / not yet implemented

- No quantitative evaluation (Precision@K / Recall@K) — recommendations are currently judged by inspection only.
- `links.csv` (IMDB/TMDB IDs) is loaded but unused — could enable poster/metadata lookups.
- No time-decay on ratings — a 2010 rating and a 2018 rating are weighted identically.
- `ALPHA` (CF vs content weight) is a fixed constant, not tuned against any metric.
