# tryon.ai — AI-Powered Virtual Try-On System

A Streamlit starter project with:
1. TF-IDF fashion product recommendations
2. An optional OpenAI-powered fashion stylist, with an offline fallback
3. A virtual try-on API adapter for a separately hosted compatible model

## Project structure

```text
tryon.ai/
├── app.py
├── recommender.py
├── stylist.py
├── tryon.py
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── products.csv
├── datasets/
│   ├── fashion_products/
│   │   └── README.md
│   ├── viton_hd/
│   │   └── README.md
│   └── deepfashion/
│       └── README.md
└── models/
    └── README.md
```

## Run locally (Mac/Linux)

```bash
cd tryon.ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Add the real product dataset

1. Download Fashion Product Images from:
   https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
2. Extract `styles.csv` to `datasets/fashion_products/styles.csv`.
3. If available, extract its `images/` directory to `datasets/fashion_products/images/`.
4. Restart Streamlit. The app uses the downloaded dataset when that CSV exists; otherwise it uses `data/products.csv`.

The common Kaggle version generally has product names, article types, colors, usage, and IDs, but not reliable sale prices. Unknown prices are kept in results, so the budget filter is only reliable when your CSV has a price column.

## Optional AI stylist

Add your key to `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Without a key, the local rule-based stylist still works. Never commit `.env` or expose your API key.

## Configure virtual try-on

The app expects a separately running endpoint that accepts multipart fields named `person_image` and `garment_image` and responds with raw image bytes. Set:

```env
TRYON_API_URL=https://your-compatible-endpoint
TRYON_API_KEY=your_optional_token
```

This request/response format is a generic adapter, not a universal standard. Change `tryon.py` to match the exact API contract of your chosen provider or model server.

VITON-HD is a dataset/research implementation, not a plug-and-play hosted endpoint. Follow its official repository setup and license:
https://github.com/shadow2496/VITON-HD

## Important limitations

- The bundled CSV is sample data to make the recommender usable immediately.
- The app does not automatically download datasets or pretrained weights.
- The try-on tab will not generate results until a compatible model endpoint is configured.
- Check dataset/model licenses and consent/privacy requirements before public or commercial deployment.
