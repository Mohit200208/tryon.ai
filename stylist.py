import os
from dotenv import load_dotenv

load_dotenv()

def get_styling_advice(request: str) -> str:
    """Use OpenAI if configured; otherwise provide a simple offline styling response."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": (
                        "You are a practical fashion stylist. Give concise outfit ideas, "
                        "color combinations, footwear/accessory suggestions, and budget-aware alternatives. "
                        "Do not claim to know current prices unless provided."
                    )},
                    {"role": "user", "content": request},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content or "Try a balanced outfit with neutral basics."
        except Exception as exc:
            return f"OpenAI request failed; using local suggestions instead. ({exc})\n\n" + _fallback(request)
    return _fallback(request)

def _fallback(request: str) -> str:
    text = request.lower()
    if any(word in text for word in ["formal", "presentation", "interview", "office"]):
        return (
            "**Smart/formal outfit**\n"
            "- Pair a white, light-blue, or pale-grey shirt with navy/charcoal trousers.\n"
            "- Choose clean loafers or formal shoes and a matching belt.\n"
            "- Keep accessories minimal; add a blazer if the occasion calls for it.\n"
            "- For a lower budget, prioritize fit and clean shoes over branded items."
        )
    if any(word in text for word in ["party", "date", "night"]):
        return (
            "**Party/date outfit**\n"
            "- Try a well-fitted dark shirt or plain tee with black/charcoal jeans.\n"
            "- Add clean sneakers or boots and one simple accessory.\n"
            "- Keep the color palette to two or three coordinated colors."
        )
    if any(word in text for word in ["sport", "gym", "run"]):
        return (
            "**Activewear**\n"
            "- Choose breathable, moisture-wicking fabric and comfortable trainers.\n"
            "- Use a simple neutral base with one accent color.\n"
            "- Prioritize movement and fit over appearance."
        )
    return (
        "**Everyday outfit idea**\n"
        "- Pair a solid tee or casual shirt with well-fitted jeans/chinos.\n"
        "- White or neutral sneakers are easy to coordinate.\n"
        "- Match one accent color across your outfit and dress for comfort.\n"
        "- Share your budget, occasion, and preferred colors for a more tailored suggestion."
    )
