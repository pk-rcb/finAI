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
debate_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)
debate_agent = debate_llm.bind_tools([get_company_metrics])

def bull_agent(state: ApplicationState):
    sys_msg = SystemMessage(content="You are a bullish stock analyst. Use your tools to look up the stock metrics. Then, write a 2-sentence argument for why the user SHOULD buy the stock based ON THE DATA.")
    response = debate_agent.invoke([sys_msg] + state["messages"])
    messages = state["messages"] + [response]
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = get_company_metrics.invoke(tool_call)
        tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"], name=tool_call["name"])
        final_arg = debate_llm.invoke([sys_msg] + messages + [tool_msg])
        return {"messages": messages + [tool_msg, final_arg]}
    return {"messages": messages}

def bear_agent(state: ApplicationState):
    sys_msg = SystemMessage(content="You are a bearish stock analyst. Use your tools to look up the stock metrics. Read the bull's argument, and write a sharp 2-sentence counter-argument focusing on valuation risks and bad data.")
    response = debate_agent.invoke([sys_msg] + state["messages"])
    messages = state["messages"] + [response]
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = get_company_metrics.invoke(tool_call)
        tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"], name=tool_call["name"])
        final_arg = debate_llm.invoke([sys_msg] + messages + [tool_msg])
        return {"messages": messages + [tool_msg, final_arg]}
    return {"messages": messages}

def judge_agent(state: ApplicationState):
    sys_msg = SystemMessage(content="""You are a ruthless, highly objective portfolio manager. 
    Read the preceding bull and bear arguments, and the live web data they cite. 
    Write a final, highly opinionated 3-sentence verdict on whether the stock is a buy, hold, or sell. 
    Do NOT include generic financial disclaimers.""")
    response = debate_llm.invoke([sys_msg] + state["messages"])
    return {"messages": state["messages"] + [response]}

# --- FUNDAMENTAL AGENT ---
def fundamental_node(state: ApplicationState):
    user_message = state["messages"][0].content
    extractor_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    ticker_query = extractor_llm.invoke([
        SystemMessage(content="""Extract the stock ticker symbol from the text. 
        1. Return ONLY the uppercase ticker symbol (e.g., AAPL).
        2. If the stock is an Indian company (e.g., Bharti Airtel, Reliance, TCS), MUST append '.NS' (e.g., BHARTIARTL.NS)."""),
        HumanMessage(content=user_message)
    ]).content.strip().upper().replace("'", "").replace('"', "")
    
    try:
        stock = yf.Ticker(ticker_query)
        info = stock.info
        def fmt(val): return f"${val:,.0f}" if isinstance(val, (int, float)) else val
        
        report = f"📈 **FUNDAMENTAL ANALYSIS: {ticker_query}**\n\n"
        report += f"**Valuation & Profitability:**\n• Trailing P/E: {info.get('trailingPE', 'N/A')}\n• Gross Margins: {info.get('grossMargins', 0) * 100:.2f}%\n"
        report += f"\n**Balance Sheet & Cash Flow:**\n• Total Cash: {fmt(info.get('totalCash', 'N/A'))}\n• Total Debt: {fmt(info.get('totalDebt', 'N/A'))}\n• Free Cash Flow: {fmt(info.get('freeCashflow', 'N/A'))}\n"

        analysis_sys = SystemMessage(content="You are a Wall Street fundamental analyst. Read the provided financial data and write a 2-sentence summary of the company's balance sheet health.")
        analysis = extractor_llm.invoke([analysis_sys, HumanMessage(content=report)])
        msg = AIMessage(content=f"🔓 *[Human Approved]*\n\n{report}\n\n**Analyst Summary:**\n{analysis.content}")
    except Exception as e:
        msg = AIMessage(content=f"Error pulling fundamental data for {ticker_query}: {str(e)}")
    return {"messages": list(state["messages"]) + [msg]}

# --- PORTFOLIO AGGREGATOR ---
def portfolio_node(state: ApplicationState):
    user_message = state["messages"][0].content # Get original user prompt
    extractor_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).with_structured_output(PortfolioExtraction)
    extracted_portfolio = extractor_llm.invoke(user_message)
    
    total_portfolio_value = 0.0
    asset_details = []
    
    for asset in extracted_portfolio.assets:
        try:
            hist = yf.Ticker(asset.ticker).history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                currency = "INR" if asset.ticker.endswith(".NS") else "USD"
                price_usd = current_price / 83.0 if currency == "INR" else current_price
                total_value = price_usd * asset.shares
                total_portfolio_value += total_value
                asset_details.append({
                    "ticker": asset.ticker, "shares": asset.shares,
                    "price_native": current_price, "currency": currency, "total_value_usd": total_value
                })
        except Exception:
            pass 
            
    report = f"\n📊 **PORTFOLIO RISK & ALLOCATION REPORT**\n**Total Estimated Value:** ${total_portfolio_value:,.2f} USD\n\n"
    for asset in asset_details:
        weight = (asset['total_value_usd'] / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        report += f"🔹 **{asset['ticker']}**: {asset['shares']} shares @ {asset['currency']} {asset['price_native']:.2f} *(Allocation: {weight:.1f}%)*\n"
        if weight > 40:
            report += f"   ⚠️ *High Concentration Risk: Consider diversifying.*\n"
            
    return {"messages": list(state["messages"]) + [AIMessage(content=report)]}

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