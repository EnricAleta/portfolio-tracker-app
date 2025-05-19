import streamlit as st
st.set_page_config(page_title="Portfolio Tracker", layout="wide")


import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
import pandas as pd
import os


from llm_utils import generate_portfolio_insight
st.write("✅ generate_portfolio_insight() imported successfully")







# -----------------------------------------------------------
# Streamlit App: Portfolio Tracker (yahooquery version)
# -----------------------------------------------------------
# Title
st.title("📈 Portfolio Tracker")

# Sidebar - Portfolio Settings
st.sidebar.header("Portfolio Settings")
st.sidebar.write("Manage your assets here.")

# Initialize session states
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "expanded_assets" not in st.session_state:
    st.session_state.expanded_assets = {}

# -----------------------------------------------------------
# Function to fetch stock/crypto data using yahooquery
# -----------------------------------------------------------
import yfinance as yf

import yfinance as yf

@st.cache_data
def get_asset_data(ticker, asset_type):
    try:
        # Check that ticker is a non-empty string
        if not ticker or not isinstance(ticker, str) or ticker.strip() == "":
            return 0.0

        ticker = asset.get("ticker", "").strip().upper()


        # Crypto needs -USD
        if asset_type == "Crypto":
            ticker = f"{ticker}-USD"

        stock = yf.Ticker(ticker)

        # Attempt to fetch history safely
        hist = stock.history(period="5d")
        if hist is not None and not hist.empty:
            if "Close" in hist.columns:
                # Use -2 to avoid same-day glitches
                return round(hist["Close"].iloc[-2], 2)

    except Exception as e:
        print(f"[WARN] Failed to fetch price for '{ticker}': {e}")

    return 0.0



# -----------------------------------------------------------
# Function to add a new asset
# -----------------------------------------------------------
def add_asset():
    asset_id = len(st.session_state.portfolio)
    st.session_state.portfolio.append({
        "id": asset_id,
        "type": "Stock",
        "name": f"Asset {asset_id + 1}",
        "ticker": "",
        "custom_name": "",  # Custom name for PE/RE/Bonds
        "investment": 0.0,
        "shares": 0.0,
        "buy_price": 0.0,  # Store buy price for crypto
        "buy_data": [],
        "sell_data": [],
        "cash_flows": {}
    })
    st.session_state.expanded_assets[asset_id] = True

# -----------------------------------------------------------
# Sidebar Button: Add New Asset
# -----------------------------------------------------------
st.sidebar.button("➕ Add New Asset", on_click=add_asset)

# -----------------------------------------------------------
# User Input for Each Asset
# -----------------------------------------------------------
for i, asset in enumerate(st.session_state.portfolio):
    with st.sidebar.expander(f"{asset['name']}"):
        # Asset Type Selection
        asset["type"] = st.selectbox(
            f"Type of {asset['name']}",
            ["Stock", "Bond", "Private Equity", "Real Estate", "Crypto", "Other"],
            index=["Stock", "Bond", "Private Equity", "Real Estate", "Crypto", "Other"].index(asset["type"]),
            key=f"type_{i}"
        )

        if asset["type"] in ["Stock", "Crypto"]:
            asset["ticker"] = st.text_input(f"Ticker:", asset["ticker"], key=f"ticker_{i}")
            
            # Buy Transactions Tracking
            buy_date = st.date_input(f"Buy Date:", datetime.today(), key=f"buy_date_{i}")
            buy_price = st.number_input(f"Buy Price per Share (€):", min_value=0.0, key=f"buy_price_{i}")
            buy_shares = st.number_input(f"Shares Purchased:", min_value=0.0, key=f"buy_shares_{i}")

            if st.button(f"Add Buy Transaction", key=f"add_buy_{i}"):
                if asset["ticker"] and buy_shares > 0:
                    asset["buy_data"].append((buy_date, round(buy_price, 2), round(buy_shares, 2)))
                    if asset["type"] == "Crypto":
                        asset["buy_price"] = round(buy_price, 2)  # Store correct buy price

            # Sell Transactions Tracking
            sell_date = st.date_input(f"Sell Date:", datetime.today(), key=f"sell_date_{i}")
            sell_price = st.number_input(f"Sell Price per Share (€):", min_value=0.0, key=f"sell_price_{i}")
            sell_shares = st.number_input(f"Shares Sold:", min_value=0.0, key=f"sell_shares_{i}")

            if st.button(f"Sell Stock", key=f"sell_stock_{i}"):
                if asset["ticker"] and sell_shares > 0:
                    asset["sell_data"].append((sell_date, round(sell_price, 2), round(sell_shares, 2)))

        elif asset["type"] in ["Private Equity", "Real Estate", "Bond"]:
            # Custom Name Input
            asset["custom_name"] = st.text_input(f"Name of Asset:", asset["custom_name"], key=f"name_{i}")

            # Alternative Investments
            asset["investment_year"] = st.number_input(
                f"Year of Purchase:",
                min_value=1900,
                max_value=2100,
                value=datetime.today().year,
                key=f"year_{i}"
            )
            asset["investment"] = st.number_input(f"Investment Amount (€):", min_value=0.0, key=f"investment_{i}")

            # Annual Cash Flow Tracking
            st.subheader(f"📊 Annual Cash Flows for {asset['type']}")
            for year in range(asset["investment_year"], datetime.today().year + 1):
                asset["cash_flows"][year] = st.number_input(f"Cash Flow in {year} (€):", value=0.0, key=f"cf_{i}_{year}")

# -----------------------------------------------------------
# Portfolio Data Processing
# Portfolio Data Processing
portfolio_data = []
total_portfolio_value = 0  # Calculate total active positions
cash_balance = 0  # Track cash from sales

for asset in st.session_state.portfolio:
    if asset["type"] in ["Stock", "Crypto"]:
        latest_price = 0.0
        ticker = asset.get("ticker", "")
        if isinstance(ticker, str) and ticker.strip():
            try:
                latest_price = get_asset_data(ticker.strip(), asset["type"])
            except Exception as e:
                st.warning(f"⚠️ Could not fetch data for {ticker.strip()}: {e}")

        total_shares_bought = sum([buy[2] for buy in asset["buy_data"]])
        total_shares_sold = sum([sell[2] for sell in asset["sell_data"]])
        total_shares = total_shares_bought - total_shares_sold

        if total_shares > 0:  # Only keep assets with open positions
            total_value = round(total_shares * latest_price, 2)
            avg_buy_price = round(
                sum([buy[1] * buy[2] for buy in asset["buy_data"]]) / max(1, total_shares_bought), 2
            ) if asset["buy_data"] else 0

            profit_loss = round(((latest_price - avg_buy_price) / avg_buy_price) * 100, 2) if avg_buy_price > 0 else 0
            total_portfolio_value += total_value

            portfolio_data.append({
                "Type": asset["type"],
                "Asset": asset["ticker"],
                "Shares": total_shares,
                "Buy Price (€)": avg_buy_price,
                "Latest Price (€)": latest_price,
                "Total Value (€)": total_value,
                "Profit/Loss (%)": profit_loss
            })

        # Add cash from sales to cash balance
        for sell in asset["sell_data"]:
            cash_balance += sell[1] * sell[2]

    else:
        asset_name = asset["custom_name"] if asset["custom_name"] else asset["name"]
        total_value = asset["investment"] + sum(asset["cash_flows"].values())
        total_portfolio_value += total_value

        portfolio_data.append({
            "Type": asset["type"],
            "Asset": asset_name,
            "Total Value (€)": total_value,
        })

# Display Portfolio Overview
st.subheader("Portfolio Overview")
df_portfolio = pd.DataFrame(portfolio_data)

if not df_portfolio.empty:
    st.dataframe(df_portfolio)

# Display Excess Cash
st.subheader("💰 Excess Cash (€)")
st.markdown(f"### {round(cash_balance, 2)} €")



# Display Total Portfolio Value
st.subheader("💰 Total Portfolio Value (€)")
st.markdown(f"### {round(total_portfolio_value + cash_balance, 2)} €")

# Total Portfolio Value Over Time
portfolio_timeline = {}
earliest_date = None

for asset in st.session_state.portfolio:
    if asset["type"] in ["Stock", "Crypto"]:
        ticker = asset["ticker"].strip()
        if not ticker:
            continue

        symbol = ticker if asset["type"] == "Stock" else f"{ticker}-USD"

        try:
            yf_ticker = yf.Ticker(symbol)
            data = yf_ticker.history(period="max")
        except Exception as e:
            st.warning(f"⚠️ Error fetching historical data for {symbol}: {e}")
            continue

        if data.empty or "Close" not in data.columns:
            st.warning(f"⚠️ No valid historical data returned for {symbol}.")
            continue

        # Clean and format price data
        data = data.reset_index()
        data = data[["Date", "Close"]]
        data.rename(columns={"Close": "Price"}, inplace=True)
        data["Date"] = pd.to_datetime(data["Date"]).dt.date  # Align to .date()

        # Exclude last 2 days to avoid data glitches
        if len(data) > 2:
            data = data.iloc[:-2]

        # Track buy/sell transactions
        transactions = {}
        for buy in asset["buy_data"]:
            date = pd.to_datetime(buy[0]).date()
            transactions[date] = transactions.get(date, 0) + buy[2]

        for sell in asset["sell_data"]:
            date = pd.to_datetime(sell[0]).date()
            transactions[date] = transactions.get(date, 0) - sell[2]

        if not transactions:
            continue

        # Build daily share count
        start_date = min(transactions.keys())
        end_date = max(data["Date"])
        date_range = pd.date_range(start=start_date, end=end_date)

        share_counts = []
        current_shares = 0

        for dt in date_range:
            date = dt.date()
            current_shares += transactions.get(date, 0)
            current_shares = max(current_shares, 0)
            share_counts.append((date, current_shares))

        df_shares = pd.DataFrame(share_counts, columns=["Date", "Shares"])

        # Merge shares with prices
        df_merged = pd.merge(data, df_shares, on="Date", how="left").fillna(method="ffill")
        df_merged["Shares"] = df_merged["Shares"].fillna(0)
        df_merged["Total Value"] = df_merged["Price"] * df_merged["Shares"]

        # Add back cash from sales
        cash_from_sales = {}
        for sell in asset["sell_data"]:
            date = pd.to_datetime(sell[0]).date()
            value = sell[1] * sell[2]
            cash_from_sales[date] = cash_from_sales.get(date, 0) + value

        df_merged["Cash Adjust"] = df_merged["Date"].apply(
            lambda d: sum(val for cash_date, val in cash_from_sales.items() if d >= cash_date)
        )
        df_merged["Total Value"] += df_merged["Cash Adjust"]

        # Add to global timeline
        for _, row in df_merged.iterrows():
            date_str = str(row["Date"])
            portfolio_timeline[date_str] = portfolio_timeline.get(date_str, 0) + row["Total Value"]

        if not df_merged.empty:
            first_date = min(df_merged["Date"])
            earliest_date = min(earliest_date, pd.Timestamp(first_date)) if earliest_date else pd.Timestamp(first_date)

    elif asset["type"] in ["Private Equity", "Real Estate", "Bond"]:
        investment_date = pd.Timestamp(f"{asset['investment_year']}-01-01")
        current_value = asset["investment"]
        portfolio_timeline[str(investment_date.date())] = (
            portfolio_timeline.get(str(investment_date.date()), 0) + current_value
        )
        earliest_date = min(earliest_date, investment_date) if earliest_date else investment_date

        last_value = current_value
        for year in range(asset["investment_year"], datetime.today().year + 1):
            date_str = f"{year}-01-01"
            if year in asset["cash_flows"]:
                last_value += asset["cash_flows"][year]
            portfolio_timeline[date_str] = portfolio_timeline.get(date_str, 0) + last_value

# Create timeline DataFrame
df_timeline = pd.DataFrame(sorted(portfolio_timeline.items()), columns=["Date", "Total Portfolio Value (€)"])
df_timeline["Date"] = pd.to_datetime(df_timeline["Date"])
df_timeline = df_timeline.sort_values("Date").reset_index(drop=True)

# Adjust for Sales — Already included above, so we skip this double-count

# Fill missing values
df_timeline["Total Portfolio Value (€)"] = df_timeline["Total Portfolio Value (€)"].ffill(limit=15)

# Filter from earliest real investment date
investment_dates = []

for asset in st.session_state.portfolio:
    if asset["type"] in ["Stock", "Crypto"]:
        investment_dates.extend([pd.to_datetime(buy[0]) for buy in asset["buy_data"]])
    elif asset["type"] in ["Private Equity", "Real Estate", "Bond"]:
        investment_dates.append(pd.to_datetime(f"{asset['investment_year']}-01-01"))

if investment_dates:
    min_investment_date = min(investment_dates)
    df_timeline = df_timeline[df_timeline["Date"] >= min_investment_date]

# Add cash balance to the final portfolio value
df_timeline["Total Portfolio Value (€)"] += cash_balance

# Apply moving average if there are cryptocurrencies
if any(asset["type"] == "Crypto" for asset in st.session_state.portfolio):
    df_timeline['Smoothed Value'] = df_timeline['Total Portfolio Value (€)'].rolling(window=7).mean()
    df_timeline['Smoothed Value'].fillna(method='bfill', inplace=True)
    st.line_chart(df_timeline.set_index("Date")["Smoothed Value"])
else:
    st.line_chart(df_timeline.set_index("Date")["Total Portfolio Value (€)"])







# -----------------------------------------------------------
# Portfolio Allocation Pie Chart
st.subheader("Portfolio Allocation by Asset Class")

# Add Excess Cash to the portfolio data
if cash_balance > 0:
    portfolio_data.append({
        "Type": "Excess Cash",
        "Asset": "Excess Cash",
        "Total Value (€)": cash_balance
    })

df_portfolio = pd.DataFrame(portfolio_data)

if not df_portfolio.empty and "Type" in df_portfolio.columns:
    df_allocation = df_portfolio.groupby("Type", as_index=False)["Total Value (€)"].sum()
    if not df_allocation.empty:
        fig_pie = px.pie(df_allocation, names="Type", values="Total Value (€)", title="Portfolio Breakdown by Asset Class")
        st.plotly_chart(fig_pie)
    else:
        st.warning("No asset data available to display the pie chart.")
else:
    st.warning("No asset data available or 'Type' column is missing.")

# Portfolio Allocation by Individual Asset
st.subheader("Portfolio Allocation by Individual Asset")

if not df_portfolio.empty and "Asset" in df_portfolio.columns:
    df_asset_allocation = df_portfolio.groupby("Asset", as_index=False)["Total Value (€)"].sum()
    if not df_asset_allocation.empty:
        fig_asset_pie = px.pie(df_asset_allocation, names="Asset", values="Total Value (€)", title="Portfolio Breakdown by Individual Asset")
        st.plotly_chart(fig_asset_pie)
    else:
        st.warning("No asset data available to display the asset breakdown pie chart.")
else:
    st.warning("No asset data available or 'Asset' column is missing.")

# Transaction History
st.subheader("📜 Transaction History")

# Buy Transactions
buy_transactions = []
for asset in st.session_state.portfolio:
    for buy in asset["buy_data"]:
        buy_transactions.append({
            "Type": asset["type"],
            "Asset": asset["custom_name"] if asset["type"] in ["Private Equity", "Real Estate", "Bond"] else asset["ticker"],
            "Buy Date": buy[0],
            "Buy Price (€)": buy[1],
            "Shares / Investment (€)": buy[2] if asset["type"] in ["Stock", "Crypto"] else asset["investment"]
        })

# Include Private Equity, Real Estate, and Bonds in Buy Transactions
for asset in st.session_state.portfolio:
    if asset["type"] in ["Private Equity", "Real Estate", "Bond"]:
        buy_transactions.append({
            "Type": asset["type"],
            "Asset": asset["custom_name"] if asset["custom_name"] else asset["name"],
            "Buy Date": f"{asset['investment_year']}-01-01",
            "Buy Price (€)": asset["investment"],
            "Shares / Investment (€)": asset["investment"]
        })

df_buy = pd.DataFrame(buy_transactions)
st.write("📌 **Buy Transactions**")
st.dataframe(df_buy)

# Sell Transactions
sell_transactions = []
for asset in st.session_state.portfolio:
    for sell in asset["sell_data"]:
        buy_prices = [b[1] for b in asset["buy_data"]]
        avg_buy_price = sum(buy_prices) / len(buy_prices) if buy_prices else 0
        sell_profit = round((sell[1] - avg_buy_price) * sell[2], 2) if avg_buy_price > 0 else 0

        sell_transactions.append({
            "Type": asset["type"],
            "Asset": asset["custom_name"] if asset["type"] in ["Private Equity", "Real Estate", "Bond"] else asset["ticker"],
            "Sell Date": sell[0],
            "Sell Price (€)": sell[1],
            "Shares Sold": sell[2],
            "Profit (€)": sell_profit
        })

# Include Private Equity, Real Estate, and Bonds in Sell Transactions
for asset in st.session_state.portfolio:
    if asset["type"] in ["Private Equity", "Real Estate", "Bond"]:
        for year, cash_flow in asset["cash_flows"].items():
            sell_transactions.append({
                "Type": asset["type"],
                "Asset": asset["custom_name"] if asset["custom_name"] else asset["name"],
                "Sell Date": f"{year}-01-01",
                "Sell Price (€)": cash_flow,
                "Shares Sold": "-",
                "Profit (€)": cash_flow  # Assuming cash flow is the profit for these asset types
            })

df_sell = pd.DataFrame(sell_transactions)
st.write("📌 **Sell Transactions**")
st.dataframe(df_sell)

# -----------------------------------------------------------
# Asset-Specific Graph (Popup)
# -----------------------------------------------------------
import yfinance as yf

if "show_asset_graph" not in st.session_state:
    st.session_state.show_asset_graph = False

st.subheader("📈 View Individual Asset Performance")

# Build dropdown asset list
asset_options = []
for a in st.session_state.portfolio:
    if a["type"] in ["Stock", "Crypto"]:
        asset_options.append(a["ticker"])
    else:
        asset_options.append(a["custom_name"] if a["custom_name"] else a["name"])

# Continue only if assets exist
if asset_options:
    selected_asset = st.selectbox("Select an asset to view its trend:", asset_options)

    if st.button("📊 Show Graph"):
        st.session_state.show_asset_graph = True

    if st.session_state.show_asset_graph and st.button("❌ Close Graph"):
        st.session_state.show_asset_graph = False
        st.rerun()

    if st.session_state.show_asset_graph and selected_asset:
        asset_data = next((x for x in st.session_state.portfolio
                           if (x.get("ticker") == selected_asset)
                           or (x.get("custom_name") == selected_asset)), None)

        if asset_data:
            # 📈 Stock or Crypto
            if asset_data["type"] in ["Stock", "Crypto"]:
                # Symbol prep
                ticker_symbol = asset_data["ticker"]
                if asset_data["type"] == "Crypto":
                    ticker_symbol += "-USD"

                ticker_obj = yf.Ticker(ticker_symbol)

                # Historical data
                if asset_data["buy_data"]:
                    first_buy_date = min(pd.to_datetime([b[0] for b in asset_data["buy_data"]]))
                    hist = ticker_obj.history(start=first_buy_date.strftime('%Y-%m-%d'))
                else:
                    hist = ticker_obj.history(period="max")

                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    hist = hist.reset_index()
                    hist = hist[['Date', 'Close']]
                    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)

                    # Shares + return
                    total_bought = sum(b[2] for b in asset_data["buy_data"])
                    total_sold = sum(s[2] for s in asset_data["sell_data"])
                    net_shares = total_bought - total_sold

                    avg_buy_price = (
                        sum(b[1] * b[2] for b in asset_data["buy_data"]) / total_bought
                        if total_bought > 0 else 0
                    )
                    latest_price = hist['Close'].iloc[-1]
                    total_return = (latest_price - avg_buy_price) * net_shares if avg_buy_price else 0
                    return_color = "green" if total_return >= 0 else "red"

                    # Plot
                    with st.expander(f"📊 {selected_asset} Performance", expanded=True):
                        st.write(f"### {selected_asset} Price Trend")
                        fig = px.line(hist, x="Date", y="Close", title=f"{selected_asset} Price Trend")
                        st.plotly_chart(fig)
                        st.markdown(
                            f"<h3>Return Since Inception: "
                            f"<span style='color:{return_color};'>{total_return:,.2f} €</span></h3>",
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No historical price data found.")

            # 🏢 PE / RE / Bond
            elif asset_data["type"] in ["Private Equity", "Real Estate", "Bond"]:
                investment_year = asset_data.get("investment_year", datetime.today().year)
                cash_flows = [-asset_data["investment"]]
                years = [investment_year]
                total_value = asset_data["investment"]
                growth = [total_value]

                for y in sorted(asset_data["cash_flows"].keys()):
                    cf = asset_data["cash_flows"][y]
                    cash_flows.append(cf)
                    years.append(y)
                    total_value += cf
                    growth.append(total_value)

                try:
                    irr = np.irr(cash_flows)
                    irr_percentage = irr * 100 if irr is not None else 0
                except:
                    irr_percentage = 0

                irr_color = "green" if irr_percentage >= 0 else "red"

                df_growth = pd.DataFrame({"Year": years, "Total Value (€)": growth})

                with st.expander(f"📊 {selected_asset} Investment Growth", expanded=True):
                    st.write(f"### {selected_asset} Value Over Time")
                    fig = px.line(df_growth, x="Year", y="Total Value (€)", title=f"{selected_asset} Investment Growth")
                    st.plotly_chart(fig)
                    st.markdown(
                        f"<h3>IRR: <span style='color:{irr_color};'>{irr_percentage:,.2f}%</span></h3>",
                        unsafe_allow_html=True
                    )

else:
    st.warning("No assets added yet. Please add one to view its trend.")




# -----------------------------
# PDF GENERATION - FULL MODULE
# -----------------------------
import streamlit as st
from fpdf import FPDF
from datetime import datetime

# Utility: Clean text to remove unsupported characters
def clean_text(text):
    return str(text).encode('latin-1', errors='ignore').decode('latin-1')

# Custom PDF class
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, clean_text(">>> Portfolio Statement"), ln=True, align="C")
        self.set_font("Arial", "", 10)
        self.cell(0, 10, clean_text(f"Generated on: {datetime.today().strftime('%Y-%m-%d')}"), ln=True, align="C")
        self.ln(5)

    def add_table(self, headers, data):
        self.set_fill_color(200, 220, 255)
        self.set_font("Arial", "B", 10)
        col_widths = [25, 22, 25, 22, 22, 25, 15, 25, 20]

        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, clean_text(header), border=1, fill=True, align='C')
        self.ln()

        self.set_font("Arial", "", 9)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 8, clean_text(item), border=1, align='C')
            self.ln()

    def add_summary(self, summary):
        self.ln(10)
        self.set_font("Arial", "B", 10)
        self.cell(0, 10, clean_text(">>> Summary"), ln=True)

        self.set_font("Arial", "", 9)
        for label, value in summary.items():
            self.cell(0, 8, clean_text(f"{label}: {value}"), ln=True)

    def add_asset_values(self, asset_values):
        self.ln(10)
        self.set_font("Arial", "B", 10)
        self.cell(0, 10, clean_text(">>> Asset Values"), ln=True)

        self.set_font("Arial", "", 9)
        for asset_type, value in asset_values.items():
            self.cell(0, 8, clean_text(f"{asset_type}: {value}"), ln=True)

# Generate the PDF
def generate_pdf_statement(portfolio, total_portfolio_value, cash_balance):
    pdf = PDF()
    pdf.add_page()

    headers = ["Ticker", "Date", "Operation", "Buy Price (EUR)", "Sell Price (EUR)",
               "Current Price (EUR)", "Shares", "Profit/Loss (EUR)", "Return (%)"]
    table_data = []
    all_returns = []
    asset_values = {"Private Equity": 0, "Real Estate": 0, "Bond": 0}

    for asset in portfolio:
        ticker = asset.get("ticker", "")
        current_price = get_asset_data(ticker, asset["type"])

        for buy in asset["buy_data"]:
            date, buy_price, shares = buy
            profit = (current_price - buy_price) * shares
            ret_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
            table_data.append([
                clean_text(ticker),
                clean_text(date.strftime("%Y-%m-%d")),
                "Buy",
                clean_text(buy_price),
                "-",
                clean_text(current_price),
                clean_text(shares),
                clean_text(round(profit, 2)),
                clean_text(round(ret_pct, 2))
            ])
            all_returns.append(profit)

        for sell in asset["sell_data"]:
            date, sell_price, shares = sell
            buy_price = asset["buy_price"] if asset["type"] == "Crypto" else "-"
            profit = (sell_price - buy_price) * shares if isinstance(buy_price, (int, float)) else 0
            ret_pct = ((sell_price - buy_price) / buy_price) * 100 if isinstance(buy_price, (float, int)) and buy_price > 0 else 0
            table_data.append([
                clean_text(ticker),
                clean_text(date.strftime("%Y-%m-%d")),
                "Sell",
                clean_text(buy_price),
                clean_text(sell_price),
                clean_text(current_price),
                clean_text(shares),
                clean_text(round(profit, 2)),
                clean_text(round(ret_pct, 2))
            ])
            all_returns.append(profit)

        if asset["type"] in ["Private Equity", "Real Estate", "Bond"]:
            asset_values[asset["type"]] += asset["investment"] + sum(asset["cash_flows"].values())

    # Add Excess Cash to the table
    if cash_balance > 0:
        table_data.append([
            "Excess Cash",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            clean_text(round(cash_balance, 2)),
            "-"
        ])

    pdf.add_table(headers, table_data)

    try:
        statement_start = min([b[0] for a in portfolio for b in a["buy_data"]])
    except ValueError:
        statement_start = datetime.today()

    summary = {
        "Total Portfolio Value (EUR)": round(total_portfolio_value + cash_balance, 2),
        "Excess Cash (EUR)": round(cash_balance, 2),
        "Total Return (EUR)": round(sum(all_returns), 2),
        "Total Return (%)": f"{round((sum(all_returns) / (total_portfolio_value - sum(all_returns)) * 100), 2) if total_portfolio_value > 0 else 0}%",
        "Number of Buys": sum(len(a["buy_data"]) for a in portfolio),
        "Number of Sells": sum(len(a["sell_data"]) for a in portfolio),
        "Assets Tracked": len(portfolio),
        "Statement Period": f"{statement_start.strftime('%Y-%m-%d')} to {datetime.today().strftime('%Y-%m-%d')}"
    }

    pdf.add_summary(summary)
    pdf.add_asset_values(asset_values)

    path = "portfolio_statement.pdf"
    pdf.output(path)
    return path

# Generate PDF
pdf_file = generate_pdf_statement(st.session_state.portfolio, total_portfolio_value, cash_balance)



# -----------------------------
# Streamlit UI Download Section
# -----------------------------

st.markdown("---")
st.subheader("Download Your PDF Statement")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if st.button("Generate and Download PDF Statement"):
    pdf_file = generate_pdf_statement(st.session_state.portfolio, total_portfolio_value, cash_balance)
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name="portfolio_statement.pdf",
            mime="application/pdf"
        )



# LLM Assistant
st.markdown("---")
st.subheader("🧠 Ask Your Portfolio Assistant")

default_qs = [
    "How many stocks and crypto assets do I have?",
    "Which asset is performing best?",
    "What is the total gain/loss in %?",
    "Give me a summary of my portfolio."
]

user_question = st.text_input("What would you like to know about your portfolio?", placeholder=default_qs[0])

if user_question:
    if not df_portfolio.empty:
        with st.spinner("Thinking..."):
            try:
                response = generate_portfolio_insight(df_portfolio.to_dict(orient="records"), user_question)
                st.markdown(f"✅ **Answer:** {response}")
            except Exception as e:
                st.error(f"LLM Error: {e}")
    else:
        st.warning("Please add assets to your portfolio first.")

if st.button("🧾 Portfolio Summary"):
    summary_response = generate_portfolio_insight(df_portfolio.to_dict(orient="records"), "Give me a short summary of my portfolio performance.")
    st.success(summary_response)

if st.button("🧠 Suggest ways to diversify my portfolio"):
    suggestion = generate_portfolio_insight(
        df_portfolio.to_dict(orient="records"),
        "What types of assets or sectors should I add to diversify better?"
    )
    st.markdown(suggestion)

if st.button("💡 Recommend 3 interesting public stocks"):
    idea = generate_portfolio_insight(
        df_portfolio.to_dict(orient="records"),
        "Can you suggest 3 interesting public stocks in trending sectors and explain why?"
    )
    st.markdown(idea)
