import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize

from data_loader import load_ratings, load_movies, load_tags, build_movie_content

ALPHA = 0.7  # weight on collaborative filtering vs content-based


def minmax(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class MovieRecommender:
    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.ratings = load_ratings()
        movies = load_movies()
        tags = load_tags()
        self.movies = build_movie_content(movies, tags)

        self.user_index = None
        self.movie_index = None
        self.user_item_matrix = None
        self.item_similarity = None
        self.content_matrix = None  # sparse, movies x vocab (TF-IDF)
        self._build()

    def _build(self):
        user_means = self.ratings.groupby("userId")["rating"].transform("mean")
        self.ratings["rating_centered"] = self.ratings["rating"] - user_means

        pivot = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating_centered"
        ).fillna(0)

        self.user_index = pivot.index.tolist()
        self.movie_index = pivot.columns.tolist()
        self.user_item_matrix = csr_matrix(pivot.values)
        self.item_similarity = cosine_similarity(self.user_item_matrix.T, dense_output=False)

        movie_id_to_content = self.movies.set_index("movieId")["content_text"]
        corpus = [movie_id_to_content.get(mid, "") for mid in self.movie_index]
        vectorizer = TfidfVectorizer(min_df=1)
        self.content_matrix = vectorizer.fit_transform(corpus)  # movies x vocab, sparse

    def recommend(self, user_id: int, top_n: int = 10):
        if user_id not in self.user_index:
            return self._popularity_fallback(top_n)

        u_idx = self.user_index.index(user_id)
        user_ratings = self.user_item_matrix[u_idx].toarray().flatten()
        rated_mask = user_ratings != 0

        # CF score: weighted sum over similar rated items
        cf_scores = np.asarray(self.item_similarity.dot(user_ratings)).flatten()

        # Content score: cosine sim between user's taste profile and every movie
        content_scores = self._content_score(user_ratings)

        final_scores = self.alpha * minmax(cf_scores) + (1 - self.alpha) * minmax(content_scores)
        final_scores[rated_mask] = -np.inf  # exclude already-seen

        top_indices = np.argsort(final_scores)[::-1][:top_n]
        movie_ids = [self.movie_index[i] for i in top_indices if final_scores[i] != -np.inf]

        return self._format_results(movie_ids)

    def _content_score(self, user_ratings: np.ndarray) -> np.ndarray:
        """Build a taste-profile vector from the user's positively-rated movies,
        weighted by how much above their average they rated it, then compare
        against every movie's content vector."""
        weights = np.clip(user_ratings, 0, None)  # only positive (above-average) ratings
        if weights.sum() == 0:
            return np.zeros(self.content_matrix.shape[0])

        profile = csr_matrix(weights).dot(self.content_matrix)  # 1 x vocab
        profile = sk_normalize(profile)
        scores = cosine_similarity(profile, self.content_matrix).flatten()
        return scores

    def _popularity_fallback(self, top_n):
        # cold-start: brand-new user with zero history -> most-rated, highest-average movies
        agg = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        agg = agg[agg["count"] >= 20].sort_values("mean", ascending=False)
        movie_ids = agg.head(top_n).index.tolist()
        return self._format_results(movie_ids)

    def _format_results(self, movie_ids):
        titles = self.movies.set_index("movieId").loc[movie_ids, "title"]
        return [{"movie_id": int(mid), "title": titles[mid]} for mid in movie_ids]


def run_interactive():
    rec = MovieRecommender()
    known_ids = set(rec.user_index)
    print(f"Loaded {len(known_ids)} users. Try an ID between {min(known_ids)}-{max(known_ids)}, "
          f"or any other number to see the cold-start fallback.")

    while True:
        raw = input("\nEnter a user ID (or 'q' to quit): ").strip()
        if raw.lower() == "q":
            break
        if not raw.isdigit():
            print("Please enter a number.")
            continue

        user_id = int(raw)
        top_n_raw = input("How many recommendations? [default 5]: ").strip()
        top_n = int(top_n_raw) if top_n_raw.isdigit() else 5

        results = rec.recommend(user_id=user_id, top_n=top_n)
        tag = "personalized" if user_id in known_ids else "cold-start (popularity fallback)"
        print(f"\n{tag} recommendations for user {user_id}:")
        for r in results:
            print(f"  - {r['title']}")


if __name__ == "__main__":
    run_interactive()
