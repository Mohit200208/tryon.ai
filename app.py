import os
from pathlib import Path
import streamlit as st
from recommender import FashionRecommender
from stylist import get_styling_advice
from tryon import run_virtual_tryon

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="tryon.ai", page_icon="👗", layout="wide")

@st.cache_resource
def load_recommender():
    return FashionRecommender(
        dataset_path=BASE_DIR / "datasets" / "fashion_products" / "styles.csv",
        fallback_path=BASE_DIR / "data" / "products.csv",
    )

st.title("👗 tryon.ai")
st.caption("AI-powered fashion recommendations, styling advice, and virtual try-on")

recommender = load_recommender()
tab1, tab2, tab3 = st.tabs(["Fashion AI", "AI Stylist", "Virtual Try-On"])

with tab1:
    st.subheader("Find your next outfit")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Describe what you're looking for", "casual blue shirt")
    with col2:
        budget = st.number_input("Maximum budget (₹)", min_value=0, value=5000, step=500)
    occasion = st.selectbox("Occasion", ["Any", "Casual", "Formal", "Party", "Sports", "Ethnic", "Travel"])
    if st.button("Recommend products", type="primary"):
        results = recommender.recommend(query, budget=budget, occasion=occasion, top_k=8)
        if results.empty:
            st.warning("No matching products found. Try a higher budget or a broader search.")
        else:
            st.success(f"Showing {len(results)} recommendations. Dataset: {recommender.source_name}")
            display_cols = [c for c in ["product_name", "category", "color", "price", "description"] if c in results.columns]
            st.dataframe(results[display_cols], use_container_width=True, hide_index=True)
            for _, item in results.head(4).iterrows():
                image_path = item.get("image_path")
                if isinstance(image_path, str) and image_path and Path(image_path).is_file():
                    st.image(image_path, width=180, caption=str(item.get("product_name", "Product")))

with tab2:
    st.subheader("Ask your AI fashion stylist")
    user_request = st.text_area(
        "Tell the stylist about your occasion, style, weather, or outfit",
        "Suggest a smart-casual outfit for a college presentation under ₹3000."
    )
    if st.button("Get styling advice"):
        with st.spinner("Creating outfit suggestions..."):
            st.markdown(get_styling_advice(user_request))

with tab3:
    st.subheader("Virtual try-on")
    st.info(
        "Upload a full/upper-body person photo and a garment image. "
        "Set TRYON_API_URL in .env to a compatible inference endpoint. "
        "The dataset itself does not perform garment transfer."
    )
    person_file = st.file_uploader("Person image", type=["png", "jpg", "jpeg"], key="person")
    garment_file = st.file_uploader("Garment image", type=["png", "jpg", "jpeg"], key="garment")
    if st.button("Generate try-on", type="primary"):
        if not person_file or not garment_file:
            st.error("Upload both images first.")
        elif not os.getenv("TRYON_API_URL"):
            st.error("TRYON_API_URL is not configured. Add a compatible model endpoint to your .env file.")
        else:
            with st.spinner("Sending images to the try-on model..."):
                try:
                    result = run_virtual_tryon(person_file.getvalue(), person_file.name,
                                               garment_file.getvalue(), garment_file.name)
                    st.image(result, caption="Virtual try-on result", use_container_width=True)
                except Exception as exc:
                    st.error(f"Try-on failed: {exc}")

with st.sidebar:
    st.header("Project status")
    st.write(f"**Recommendation data:** {recommender.source_name}")
    st.write("**Stylist:** local fallback + optional OpenAI API")
    st.write("**Virtual try-on:** requires a configured compatible endpoint")
    st.markdown("[Dataset setup guide](README.md)")
