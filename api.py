"""
FastAPI service for movie recommendations.
Run: uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from recommender import MovieRecommender

app = FastAPI(title="Movie Recommendation API")

# Precompute once at startup, not per-request
recommender = MovieRecommender()


@app.get("/")
def root():
    return {"status": "ok", "message": "Movie Recommendation API"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    if top_n < 1 or top_n > 50:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 50")
    results = recommender.recommend(user_id=user_id, top_n=top_n)
    # results already exclude movies the user has rated; alpha controls CF vs content-based weight
    return {"user_id": user_id, "recommendations": results}
