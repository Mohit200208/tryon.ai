from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FashionRecommender:
    """Loads a Kaggle Fashion Product Images styles.csv or the bundled sample CSV."""

    def __init__(self, dataset_path, fallback_path):
        self.dataset_path = Path(dataset_path)
        self.fallback_path = Path(fallback_path)
        self.df = self._load_data()
        self.source_name = "Fashion Product Images dataset" if self.dataset_path.exists() else "bundled sample CSV"
        self.df = self._normalise(self.df)
        self.df["search_text"] = (
            self.df["product_name"].fillna("").astype(str) + " " +
            self.df["category"].fillna("").astype(str) + " " +
            self.df["color"].fillna("").astype(str) + " " +
            self.df["style"].fillna("").astype(str) + " " +
            self.df["occasion"].fillna("").astype(str) + " " +
            self.df["description"].fillna("").astype(str)
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"])

    def _load_data(self):
        if self.dataset_path.exists():
            try:
                return pd.read_csv(self.dataset_path, low_memory=False)
            except Exception:
                pass
        return pd.read_csv(self.fallback_path)

    @staticmethod
    def _normalise(df):
        # Support the common Kaggle "Fashion Product Images" styles.csv columns.
        aliases = {
            "product_name": ["product_name", "productDisplayName", "name", "title"],
            "category": ["category", "articleType", "subCategory", "masterCategory"],
            "color": ["color", "baseColour", "colour"],
            "style": ["style", "gender", "masterCategory"],
            "occasion": ["occasion", "usage", "season"],
            "description": ["description", "productDisplayName", "name"],
            "price": ["price", "selling_price", "price_inr"],
            "id": ["id", "product_id", "productId"],
        }
        out = pd.DataFrame(index=df.index)
        for target, choices in aliases.items():
            source = next((c for c in choices if c in df.columns), None)
            out[target] = df[source] if source else ""
        out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0)
        out["product_name"] = out["product_name"].fillna("Unnamed fashion item").astype(str)
        out["category"] = out["category"].fillna("").astype(str)
        out["color"] = out["color"].fillna("").astype(str)
        out["style"] = out["style"].fillna("").astype(str)
        out["occasion"] = out["occasion"].fillna("").astype(str)
        out["description"] = out["description"].fillna("").astype(str)

        # Optional local image matching: images/<id>.jpg is common in the Kaggle dataset.
        base = Path(__file__).resolve().parent / "datasets" / "fashion_products"
        image_paths = []
        for product_id in out["id"].astype(str):
            candidates = [base / "images" / f"{product_id}.jpg",
                          base / "images" / f"{product_id}.png",
                          base / f"{product_id}.jpg"]
            image_paths.append(str(next((p for p in candidates if p.is_file()), "")))
        out["image_path"] = image_paths
        return out.drop_duplicates(subset=["product_name", "category", "color"]).reset_index(drop=True)

    def recommend(self, query, budget=5000, occasion="Any", top_k=8):
        if not query.strip():
            query = "fashion"
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        result = self.df.copy()
        result["similarity"] = scores
        if budget and budget > 0:
            # Real-world Kaggle dataset may not have prices; don't filter it all out.
            priced = result["price"] > 0
            if priced.any():
                result = result[(~priced) | (result["price"] <= budget)]
        if occasion != "Any":
            mask = result["occasion"].str.contains(re.escape(occasion), case=False, na=False)
            if mask.any():
                result = result[mask]
        return result.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)
