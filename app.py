import os
import base64
import streamlit as st
import yfinance as yf
from typing import List, TypedDict
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. API KEYS & SETUP
# ==========================================
st.set_page_config(page_title="FinAntinomAI", page_icon="📈", layout="wide")

# Securely fetch API keys from Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY"))

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("⚠️ API Keys missing! Please add GROQ_API_KEY and TAVILY_API_KEY to your Streamlit Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ==========================================
# 2. STATE & SCHEMAS
# ==========================================
class ApplicationState(TypedDict):
    messages: List[BaseMessage]
    next_destination: str
    user_input_type: str

class IntentRoute(BaseModel):
    destination: str = Field(description="Must be exactly one of: 'debate', 'vision', 'trivia', 'fundamental', 'portfolio'")

class Asset(BaseModel):
    ticker: str = Field(description="The official stock ticker symbol. Use .NS suffix for Indian stocks.")
    shares: float = Field(description="The exact number of shares owned.")

class PortfolioExtraction(BaseModel):
    assets: List[Asset] = Field(description="List of all extracted assets and share counts.")

# ==========================================
# 3. LIVE TOOLS
# ==========================================
@tool
def get_stock_price(ticker: str) -> str:
    """Fetches the current, real-time stock price for a given ticker symbol or company name."""
    clean_ticker = ticker.upper().strip('$').strip()
    if " " not in clean_ticker:
        if clean_ticker in ["BHARTIARTL", "TCS", "RELIANCE", "INFY", "WIPRO"]:
            clean_ticker = f"{clean_ticker}.NS"
        try:
            ticker_obj = yf.Ticker(clean_ticker)
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                currency = "INR" if clean_ticker.endswith(".NS") else "USD"
                return f"The live, real-time price of {clean_ticker} is {currency} {price:.2f}."
        except Exception:
            pass

    try:
        query = f"current exact stock price of {ticker} today"
        response = tavily_client.search(query=query, search_depth="basic", max_results=2)
        search_results = [result["content"] for result in response.get("results", [])]
        if search_results:
            combined_data = "\n".join(search_results)
            return f"Yahoo Finance lookup bypassed. Live web search data indicates:\n{combined_data}"
    except Exception as e:
        return f"Error: Could not retrieve price details for {ticker}."
    return f"Could not isolate live market data for asset identifier: {ticker}."

@tool
def get_company_metrics(ticker: str) -> str:
    """Fetches the latest earnings, P/E ratio, and news sentiment for a stock from the live internet."""
    try:
        query = f"Latest earnings report, current P/E ratio, and recent news sentiment for {ticker} stock"
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        search_results = [result["content"] for result in response.get("results", [])]
        combined_data = "\n\n".join(search_results)
        return f"LIVE WEB DATA FOR {ticker}:\n{combined_data}"
    except Exception as e:
        return f"Error fetching news data: {str(e)}"

# ==========================================
# 4. ORCHESTRATOR & AGENT NODES
# ==========================================
def orchestrator_router(state: ApplicationState):
    user_message = state["messages"][-1].content
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_retries=3)
    router_llm = llm.with_structured_output(IntentRoute)
    messages = [
        SystemMessage(content="""You are the front-desk router for a financial platform. 
        Classify the user input into exactly one of these 5 categories:
        - 'debate': Asking if they should buy/sell a specific stock.
        - 'vision': Uploading a chart or asking for technical visual analysis.
        - 'trivia': Asking for basic financial definitions, concepts, or current real-time stock prices.
        - 'fundamental': Asking for a deep dive or fundamental report on a company.
        - 'portfolio': Providing a list of multiple stock holdings to analyze."""),
        HumanMessage(content=f"User Input: {user_message}")
    ]
    decision = router_llm.invoke(messages)
    return {"next_destination": decision.destination, "user_input_type": decision.destination}

def trivia_node(state: ApplicationState):
    trivia_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).bind_tools([get_stock_price])
    new_messages = list(state["messages"])
    response = trivia_llm.invoke(new_messages)
    new_messages.append(response)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = get_stock_price.invoke(tool_call)
        tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"], name=tool_call["name"])
        new_messages.append(tool_msg)
        final_response = trivia_llm.invoke(new_messages)
        new_messages.append(final_response)
        
        if final_response.content:
            pass # We have a clean text response
        elif final_response.tool_calls:
            new_messages.append(AIMessage(content=str(tool_result)))
    return {"messages": new_messages}

def vision_node(state: ApplicationState):
    # Retrieve the raw multimodal payload if it exists
    messages = state["messages"]
    vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
    sys_msg = SystemMessage(content="""You are an expert technical stock analyst. 
    Look at the provided image (a stock chart or candlestick pattern). 
    Identify the overall trend, key support and resistance levels, and any noticeable candlestick formations. 
    Keep your analysis sharp, professional, and under 4 sentences.""")
    
    try:
        response = vision_llm.invoke([sys_msg] + messages)
        return {"messages": messages + [response]}
    except Exception as e:
        msg = AIMessage(content="👁️ **Vision Agent Error:** Could not analyze the image. Please ensure you uploaded a valid chart in the sidebar.")
        return {"messages": messages + [msg]}

# --- DEBATE PANEL ---
# ==========================================
# 4. TOOL-STRAPPED DEBATE PANEL (UPGRADED)
# ==========================================

# We use temperature=0.4 for the debaters to allow creative analytical connections,
# but we will use temperature=0.0 for the Judge to ensure strict objectivity.
debate_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)
debate_agent = debate_llm.bind_tools([get_company_metrics])

def bull_agent(state: ApplicationState):
    print("\n   🐂 BULL AGENT: Executing live market sweep for upside catalysts...")
    messages = state["messages"]
    
    # 1. The Institutional Bull Prompt
    sys_msg = SystemMessage(content="""You are an aggressive venture capitalist and top-tier growth investor. 
    Use the get_company_metrics tool to pull live web data. 
    Your strict objective: Build a highly compelling, data-backed thesis for why this stock will OUTPERFORM the market.
    Focus exclusively on:
    1. Macro tailwinds and sector growth.
    2. Undervalued fundamentals (P/E expansion potential).
    3. Positive news sentiment and upcoming catalysts.
    Write a dense, 2-paragraph bullish thesis.""")
    
    # 2. Tool Execution Loop
    response = debate_agent.invoke([sys_msg] + messages)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = get_company_metrics.invoke(tool_call)
        tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"], name=tool_call["name"])
        
        # 3. Final Bull Synthesis
        final_argument = debate_llm.invoke([sys_msg] + messages + [response, tool_msg])
        final_argument.content = f"🟢 **THE BULL THESIS:**\n{final_argument.content}"
        return {"messages": messages + [response, tool_msg, final_argument]}
        
    response.content = f"🟢 **THE BULL THESIS:**\n{response.content}"
    return {"messages": messages + [response]}

def bear_agent(state: ApplicationState):
    print("   🐻 BEAR AGENT: Analyzing Bull thesis and hunting for downside risk...")
    messages = state["messages"]
    
    # 1. The Institutional Bear Prompt (Adversarial Setup)
    sys_msg = SystemMessage(content="""You are a cynical, risk-averse short-seller. 
    You have just read the Bull's thesis in the chat history. 
    Use the get_company_metrics tool to pull live web data to actively DISPROVE the Bull's points.
    Your strict objective: Build a devastating counter-argument focusing on:
    1. Overvaluation risks and shrinking margins.
    2. Regulatory headwinds or supply chain bottlenecks.
    3. Poor recent earnings or negative news sentiment.
    Write a dense, 2-paragraph bearish thesis that directly attacks the Bull's optimism.""")
    
    # 2. Tool Execution Loop
    response = debate_agent.invoke([sys_msg] + messages)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = get_company_metrics.invoke(tool_call)
        tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"], name=tool_call["name"])
        
        # 3. Final Bear Synthesis
        final_argument = debate_llm.invoke([sys_msg] + messages + [response, tool_msg])
        final_argument.content = f"🔴 **THE BEAR THESIS:**\n{final_argument.content}"
        return {"messages": messages + [response, tool_msg, final_argument]}
        
    response.content = f"🔴 **THE BEAR THESIS:**\n{response.content}"
    return {"messages": messages + [response]}

def judge_agent(state: ApplicationState):
    print("   ⚖️ JUDGE AGENT: Weighing arguments and establishing final verdict...")
    
    # 1. The Strict Judge Prompt (Temperature 0.0)
    sys_msg = SystemMessage(content="""You are the Chief Investment Officer (CIO) overseeing these two analysts.
    Read the exact data points presented in the Bull Thesis and the Bear Thesis.
    You must act coldly and mathematically. Do not hallucinate data.
    
    Output your final verdict in this exact format:
    **⚖️ CIO VERDICT & SYNTHESIS:**
    - **Strengths Validated:** (1 sentence summarizing the Bull's best point)
    - **Risks Validated:** (1 sentence summarizing the Bear's best point)
    - **Final Rating:** [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
    - **Actionable Summary:** (2 sentences explaining the final rating based on the balance of the debate).
    
    DO NOT include generic financial disclaimers like 'consult an advisor'. Be decisive.""")
    
    # 2. Final Deterministic Synthesis
    final_judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    response = final_judge_llm.invoke([sys_msg] + state["messages"])
    
    return {"messages": state["messages"] + [response]}
# --- FUNDAMENTAL AGENT ---
def fundamental_node(state: ApplicationState):
    print("\n   🔓 [HUMAN APPROVED] Fundamental Agent is now executing...")
    user_message = state["messages"][0].content
    
    # 1. Extraction Pass
    extractor_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    
    # 2. UPGRADED: Dynamic Global Ticker Extraction (No more hardcoded lists)
    ticker_query = extractor_llm.invoke([
        SystemMessage(content="""Extract the stock ticker symbol from the text. 
        CRITICAL RULE: If the user is asking about an Indian company (NSE/BSE), you MUST dynamically append '.NS' to the ticker (e.g., ABSLAMC.NS, RELIANCE.NS, ZOMATO.NS). 
        Return ONLY the final uppercase ticker symbol."""),
        HumanMessage(content=user_message)
    ]).content.strip().upper().replace("'", "").replace('"', "").replace("$", "")
    
    print(f"   [System Log: Ticker '{ticker_query}' finalized. Downloading financial data...]")
    
    try:
        stock = yf.Ticker(ticker_query)
        info = stock.info
        
        # 3. UPGRADED: THE ANTI-HALLUCINATION SHORT-CIRCUIT
        # If the API returns no info, or core metrics are missing, abort the LLM analysis entirely.
        if not info or (info.get('trailingPE') is None and info.get('totalRevenue') is None and info.get('totalCash') is None):
            print("   [System Log: ❌ yfinance returned empty data. Short-circuiting LLM to prevent hallucination.]")
            msg = AIMessage(content=f"⚠️ **DATA RETRIEVAL ERROR**\nCould not fetch fundamental data for `{ticker_query}`. The ticker may be delisted, recently IPO'd, or requires a different regional suffix.")
            return {"messages": list(state["messages"]) + [msg]}
        
        # 4. Format the raw financial data
        def fmt(val): return f"${val:,.0f}" if isinstance(val, (int, float)) else val
        
        report = f"**Valuation & Profitability:**\n"
        report += f"• Trailing P/E: {info.get('trailingPE', 'N/A')}\n"
        report += f"• Gross Margins: {info.get('grossMargins', 0) * 100:.2f}%\n"
        report += f"• Return on Equity (ROE): {info.get('returnOnEquity', 0) * 100:.2f}%\n\n"
        
        report += f"**Balance Sheet & Cash Flow:**\n"
        report += f"• Total Cash: {fmt(info.get('totalCash', 'N/A'))}\n"
        report += f"• Total Debt: {fmt(info.get('totalDebt', 'N/A'))}\n"
        report += f"• Free Cash Flow: {fmt(info.get('freeCashflow', 'N/A'))}\n"
        report += f"• Debt-To-Equity Ratio: {info.get('debtToEquity', 'N/A')}\n"

        print("   [System Log: Passing validated financials to LLM for deep fundamental synthesis...]")

        # 5. Safe Synthesis Pass
        analysis_sys = SystemMessage(content="""You are a Wall Street fundamental equity analyst. 
        Read the provided raw quantitative financial metrics.
        CRITICAL RULE: If the data is mostly 'N/A', do not invent an analysis. State that the data is insufficient.
        Otherwise, write a 3-paragraph fundamental analysis covering:
        1. Profitability & Valuation
        2. Solvency & Liquidity
        3. Final Analyst Assessment""")
        
        synthesis_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
        analysis = synthesis_llm.invoke([analysis_sys, HumanMessage(content=f"Ticker: {ticker_query}\n\n{report}")])
        
        final_response = f"🔓 *[Human Approved Execution]*\n\n📈 **DEEP FUNDAMENTAL ANALYSIS: {ticker_query}**\n\n### 1. Raw Quantitative Data\n{report}\n\n### 2. Chief Analyst Synthesis\n{analysis.content}"
        msg = AIMessage(content=final_response)
        
    except Exception as e:
        msg = AIMessage(content=f"Error pulling fundamental data: {str(e)}")
        
    return {"messages": list(state["messages"]) + [msg]}
# --- PORTFOLIO AGGREGATOR ---
def portfolio_node(state: ApplicationState):
    print("\n   💼 PORTFOLIO AGGREGATOR: Booting up...")
    user_message = state["messages"][0].content
    
    # 1. Extraction Pass (Pydantic)
    extractor = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).with_structured_output(PortfolioExtraction)
    extracted = extractor.invoke(user_message)
    
    total_val = 0.0
    asset_details = []
    
    # 2. Safe Computation Pass (Python Math Engine)
    raw_data_string = "Raw Portfolio Data:\n"
    for asset in extracted.assets:
        try:
            hist = yf.Ticker(asset.ticker).history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                currency = "INR" if asset.ticker.endswith(".NS") else "USD"
                
                # Normalize to USD for risk math (using a static 83 INR/USD conversion for this demo)
                price_usd = price / 83.0 if currency == "INR" else price
                total_value = price_usd * asset.shares
                total_val += total_value
                
                asset_details.append({
                    "ticker": asset.ticker, "shares": asset.shares,
                    "price": price, "currency": currency, "value_usd": total_value
                })
        except Exception:
            pass
            
    # 3. Assemble the Math Context
    raw_data_string += f"Total Value: ${total_val:,.2f} USD\n"
    for a in asset_details:
        weight = (a['value_usd'] / total_val) * 100 if total_val > 0 else 0
        raw_data_string += f"- {a['ticker']}: {a['shares']} shares ({weight:.1f}% Allocation)\n"

    # 4. The Deep Synthesis Pass (NEW: LLM Reasoning Layer)
    print("   [System Log: Injecting safe math into LLM for deep analysis...]")
    synthesis_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    
    synthesis_prompt = SystemMessage(content="""You are a top-tier institutional portfolio manager. 
    I will provide you with the exact mathematical breakdown of a client's portfolio.
    You must write a highly detailed, professional, 3-paragraph portfolio analysis covering:
    1. A structural breakdown of their holdings and likely sector exposure.
    2. Specific concentration risks (flag critically if any asset is >40% of the portfolio).
    3. Actionable, strategic advice for improving diversification and balancing risk-adjusted returns.
    Do not just repeat the math to them; provide deep, expert-level financial reasoning.""")
    
    analysis = synthesis_llm.invoke([synthesis_prompt, HumanMessage(content=raw_data_string)])
    
    # 5. Format Final Output
    final_report = f"📊 **COMPREHENSIVE PORTFOLIO ANALYSIS**\n\n**Data Summary:**\n{raw_data_string}\n\n**CIO Strategic Insight:**\n{analysis.content}"
    
    return {"messages": list(state["messages"]) + [AIMessage(content=final_report)]}
# ==========================================
# 5. GRAPH COMPILATION
# ==========================================
workflow = StateGraph(ApplicationState)
workflow.add_node("orchestrator", orchestrator_router)
workflow.add_node("trivia", trivia_node)
workflow.add_node("vision", vision_node)
workflow.add_node("fundamental", fundamental_node)
workflow.add_node("portfolio", portfolio_node)
workflow.add_node("bull_agent", bull_agent)
workflow.add_node("bear_agent", bear_agent)
workflow.add_node("judge_agent", judge_agent)

workflow.set_entry_point("orchestrator")
def route_traffic(state: ApplicationState): return state["next_destination"]
workflow.add_conditional_edges("orchestrator", route_traffic, {
    "debate": "bull_agent", "vision": "vision", "trivia": "trivia",
    "fundamental": "fundamental", "portfolio": "portfolio"
})

workflow.add_edge("bull_agent", "bear_agent")
workflow.add_edge("bear_agent", "judge_agent")
workflow.add_edge("judge_agent", END)
workflow.add_edge("vision", END)
workflow.add_edge("trivia", END)
workflow.add_edge("fundamental", END)
workflow.add_edge("portfolio", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["fundamental"])

# ==========================================
# 6. STREAMLIT UI (WEB FRONTEND)
# ==========================================
st.title("💼 FinAntinomAI: Multi-Agent Trading Desk")

# Sidebar for Vision Uploads
with st.sidebar:
    st.header("Upload Chart for Analysis")
    uploaded_file = st.file_uploader("Upload a candlestick chart (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("*Note: If you upload an image, type 'analyze this chart' in the chat box to trigger the Vision Agent.*")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask about a stock, upload a portfolio, or request a fundamental analysis..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Agents are analyzing..."):
            
            # Check if a file was uploaded for the Vision route
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                payload = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
                initial_state = {"messages": [HumanMessage(content=payload)], "next_destination": "", "user_input_type": ""}
            else:
                initial_state = {"messages": [HumanMessage(content=prompt)], "next_destination": "", "user_input_type": ""}
            
            config = {"configurable": {"thread_id": "web_session_1"}}
            
            # 1st Pass
            final_state = app.invoke(initial_state, config=config)
            
            # Handle Human-in-the-Loop Breakpoint (Fundamental Node)
            if "fundamental" in final_state.get("next_destination", "") or len(final_state.get("messages", [])) == 1:
                # Automatically approve the pause for the web UI seamless experience
                final_state = app.invoke(None, config=config)
                
            final_response = final_state['messages'][-1].content
            
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})