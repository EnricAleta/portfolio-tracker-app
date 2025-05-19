# 📈 Portfolio Tracker App

## 🎯 Project Overview

This Streamlit app was designed to bring order, clarity, and decision-making power to personal investing. It allows users to track multiple asset classes — from **public stocks and crypto** to **private equity, real estate, and bonds** — in one clean dashboard.

Beyond visualization, the app also includes:
- A full **transaction history** with profit/loss tracking
- A **timeline view** of portfolio value growth
- A smart **PDF statement generator**
- An integrated **LLM assistant** that answers natural-language portfolio questions and provides diversification advice

This is not a mockup — the tool is built for real, multi-asset portfolios.

---

## 🛠️ Features

- Add public and private assets (Stock, Crypto, PE, RE, Bonds)
- Track buy/sell operations, average prices, and value over time
- Monitor **excess cash**, total return, and asset-specific growth
- View allocation by asset class and individual instruments
- Generate downloadable **PDF statements** of historical performance
- Ask questions like:
  - “What’s my best performing asset?”
  - “How can I diversify this portfolio?”
  - “Give me a summary of my crypto returns.”
- Fully built with Streamlit, Plotly, Yahoo Finance, and OpenAI API

---

## 💡 Motivation

This project reflects a personal need: I wanted a real-time dashboard to monitor a complex investment portfolio across traditional and alternative assets, **not just track tickers**.

Inspired by how institutional dashboards work — but made accessible for individuals — I created a tool that does more than track: it **analyzes, summarizes, and advises**.

---

## 🧠 Architecture

- `app - final.py` — Main Streamlit interface
- `llm_utils.py` — Logic to call a custom financial assistant using OpenAI
- Uses `yfinance` to pull real-time data
- Modular logic for tracking, updating, and calculating returns
- Interactive charts using Plotly
- PDF reports with `fpdf` library
- LLM responses are tuned with market context (March 2025)

---

## 📁 Structure

- `app - final.py` — Complete Streamlit app code
- `llm_utils.py` — Insight generation logic for the financial assistant

---

## 🧾 Key Outputs

- Portfolio value over time (adjusted for cash flows and sales)
- Return per asset with color-coded performance
- Transaction log of all buy/sell operations
- IRR for private equity and real estate assets
- PDF summary report (downloadable)
- Real-time insights via GPT-powered prompts

---

## 👨‍💼 About Me

I'm Enric Aletà, a finance and data enthusiast pursuing my MSc in Business Analytics. I built this app as part of a broader effort to combine **investment logic** with **smart automation** — and to explore how tools like Streamlit and LLMs can simplify financial decision-making.

Let’s connect on [LinkedIn](https://www.linkedin.com/in/enricaletacumellas/) or reach out if you’d like to try the app or collaborate on enhancements.


