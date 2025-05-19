import streamlit as st
import openai

# Load OpenAI key from .streamlit/secrets.toml
openai.api_key = st.secrets["openai"]["api_key"]

def generate_portfolio_insight(data, question):
    print("🔍 LLM called with question:", question)

    system_prompt = (
        "You are a professional financial assistant. Your role is to analyze personal investment portfolios and provide "
        "valuable insights, recommendations, and diversification suggestions.\n\n"
        "The user's portfolio may contain Stocks, Crypto, Bonds, Private Equity, or Real Estate. If the portfolio lacks "
        "diversification, gently suggest other asset classes and explain their advantages.\n\n"
        "Based on asset types, sectors, and performance, highlight:\n"
        "- which assets are performing best or worst,\n"
        "- overall risk exposure,\n"
        "- and possible next steps for improvement.\n\n"
        "If the user only has public market assets (e.g., stocks/crypto), suggest exploring Private Equity or Real Estate "
        "with short, clear advantages (e.g., stability, passive income, low correlation).\n\n"
        "Feel free to mention 2–3 public stock suggestions in trending sectors (e.g., AI, healthcare, clean energy), "
        "based on the current market environment (March 2025). Make these ideas brief and note they are suggestions, not financial advice.\n\n"
        "Always be concise, insightful, and avoid generic advice."
    )

    market_context = (
        "Recent market trends (March 2025): AI and semiconductors remain strong. Clean energy stocks are rebounding. "
        "Healthcare and infrastructure are seen as defensive plays. Interest rates are high, so bonds may offer good yields. "
        "Crypto is volatile, with renewed attention on Bitcoin ETFs."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Market Context: {market_context}"},
        {"role": "user", "content": f"Portfolio Data: {data}"},
        {"role": "user", "content": f"Question: {question}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5,
            max_tokens=400
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ Error generating response: {e}"