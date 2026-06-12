import os
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
# 1. APPLICATION STATE
# ==========================================
class ApplicationState(TypedDict):
    messages: List[BaseMessage]
    next_destination: str
    user_input_type: str

# ==========================================
# 2. LIVE TOOLS (TAVILY & YFINANCE)
# ==========================================
# In production, these should be securely pulled from st.secrets
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def get_stock_price(ticker: str) -> str:
    """Fetches the current, real-time stock price for a given ticker symbol."""
    clean_ticker = ticker.upper().strip('$').strip()
    
    # yfinance logic with Indian stock handling
    if " " not in clean_ticker:
        if clean_ticker in ["BHARTIARTL", "TCS", "RELIANCE", "INFY", "WIPRO"]:
            clean_ticker = f"{clean_ticker}.NS"
        try:
            ticker_obj = yf.Ticker(clean_ticker)
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                currency = "INR" if clean_ticker.endswith(".NS") else "USD"
                return f"The live price of {clean_ticker} is {currency} {price:.2f}."
        except Exception:
            pass

    # Tavily Fallback
    try:
        query = f"current exact stock price of {ticker} today"
        response = tavily_client.search(query=query, search_depth="basic", max_results=2)
        search_results = [result["content"] for result in response.get("results", [])]
        if search_results:
            return "Live web search data indicates:\n" + "\n".join(search_results)
    except Exception as e:
        return f"Error fetching price: {str(e)}"
        
    return f"Could not isolate live market data for: {ticker}."

@tool
def get_company_metrics(ticker: str) -> str:
    """Fetches the latest earnings, P/E ratio, and news sentiment for a stock from the live internet."""
    try:
        query = f"Latest earnings report, current P/E ratio, and recent news sentiment for {ticker} stock"
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        search_results = [result["content"] for result in response.get("results", [])]
        return f"LIVE WEB DATA FOR {ticker}:\n" + "\n\n".join(search_results)
    except Exception as e:
        return f"Error fetching news data: {str(e)}"

# ==========================================
# 3. ROUTER & WORKER NODES
# ==========================================
class IntentRoute(BaseModel):
    destination: str = Field(description="Must be exactly one of: 'debate', 'vision', 'trivia', 'fundamental', 'portfolio'")

def orchestrator_router(state: ApplicationState):
    user_message = state["messages"][-1].content
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    router_llm = llm.with_structured_output(IntentRoute)
    
    messages = [
        SystemMessage(content="""You are the front-desk router for a financial platform. 
        Classify the user input into exactly one of these 5 categories:
        - 'debate': Asking if they should buy/sell a specific stock.
        - 'vision': Uploading a chart or asking for technical visual analysis.
        - 'trivia': Asking for basic financial definitions or current stock prices.
        - 'fundamental': Asking for a deep dive or fundamental report on a company balance sheet.
        - 'portfolio': Providing a list of multiple stock holdings to analyze."""),
        HumanMessage(content=f"User Input: {user_message}")
    ]
    
    decision = router_llm.invoke(messages)
    return {"next_destination": decision.destination, "user_input_type": decision.destination}

def trivia_node(state: ApplicationState):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).bind_tools([get_stock_price])
    messages = state["messages"]
    response = llm.invoke(messages)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = get_stock_price.invoke(tool_call)
        tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"], name=tool_call["name"])
        final_response = llm.invoke(messages + [response, tool_msg])
        return {"messages": messages + [response, tool_msg, final_response]}
        
    return {"messages": messages + [response]}

def vision_node(state: ApplicationState):
    # Standard text stub for Streamlit (to avoid complex base64 file uploads in the basic UI)
    msg = AIMessage(content="👁️ **Vision Agent:** To analyze charts on the web, please upload the image file. *(Note: Multimodal file parsing requires the advanced Streamlit file_uploader widget).*")
    return {"messages": state["messages"] + [msg]}

# --- DEBATE PANEL ---
debate_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4).bind_tools([get_company_metrics])

def bull_agent(state: ApplicationState):
    sys_msg = SystemMessage(content="You are a bullish stock analyst. Use tools to look up metrics, then write a 2-sentence argument for why the user SHOULD buy the stock.")
    response = debate_llm.invoke([sys_msg] + state["messages"])
    # Simplified without tool execution loop for UI speed
    return {"messages": state["messages"] + [response]}

def bear_agent(state: ApplicationState):
    sys_msg = SystemMessage(content="You are a bearish stock analyst. Read the bull's argument, and write a sharp 2-sentence counter-argument focusing on valuation risks.")
    response = debate_llm.invoke([sys_msg] + state["messages"])
    return {"messages": state["messages"] + [response]}

def judge_agent(state: ApplicationState):
    sys_msg = SystemMessage(content="You are an objective portfolio manager. Read the bull and bear arguments. Write a final, definitive 3-sentence verdict on whether the stock is a buy, hold, or sell.")
    response = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).invoke([sys_msg] + state["messages"])
    return {"messages": state["messages"] + [response]}

# --- FUNDAMENTAL AGENT (HITL + YFINANCE) ---
def fundamental_node(state: ApplicationState):
    user_message = state["messages"][0].content
    extractor_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    raw_ticker = extractor_llm.invoke([
        SystemMessage(content="Extract the stock ticker symbol from the text. Return ONLY the uppercase ticker symbol (e.g., AAPL, BHARTIARTL)."),
        HumanMessage(content=user_message)
    ]).content.strip().upper().replace("'", "").replace('"', "").replace("$", "")
    
    indian_stocks = ["BHARTIAIRTEL", "BHARTIARTL", "TCS", "RELIANCE", "INFY", "WIPRO", "HDFCBANK"]
    ticker_query = f"{'BHARTIARTL' if raw_ticker == 'BHARTIAIRTEL' else raw_ticker}.NS" if raw_ticker in indian_stocks else raw_ticker
    
    try:
        stock = yf.Ticker(ticker_query)
        info = stock.info
        def fmt(val): return f"${val:,.0f}" if isinstance(val, (int, float)) else val
        
        report = f"📈 **FUNDAMENTAL ANALYSIS: {ticker_query}**\n\n"
        report += f"**Valuation & Profitability:**\n• Trailing P/E: {info.get('trailingPE', 'N/A')}\n• Gross Margins: {info.get('grossMargins', 0) * 100:.2f}%\n"
        report += f"\n**Balance Sheet & Cash Flow:**\n• Total Cash: {fmt(info.get('totalCash', 'N/A'))}\n• Total Debt: {fmt(info.get('totalDebt', 'N/A'))}\n"
        
        analysis = extractor_llm.invoke([SystemMessage(content="Write a 2-sentence summary of this balance sheet health."), HumanMessage(content=report)])
        msg = AIMessage(content=f"🔓 *[Human Approved]*\n\n{report}\n**Analyst Summary:**\n{analysis.content}")
    except Exception as e:
        msg = AIMessage(content=f"Error pulling fundamental data for {ticker_query}: {str(e)}")
        
    return {"messages": list(state["messages"]) + [msg]}

# --- PORTFOLIO AGGREGATOR ---
class Asset(BaseModel):
    ticker: str
    shares: float

class PortfolioExtraction(BaseModel):
    assets: List[Asset]

def portfolio_node(state: ApplicationState):
    user_message = state["messages"][-1].content
    extractor = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0).with_structured_output(PortfolioExtraction)
    extracted = extractor.invoke(user_message)
    
    total_val = 0.0
    report = "💼 **PORTFOLIO RISK REPORT**\n\n"
    
    for asset in extracted.assets:
        try:
            hist = yf.Ticker(asset.ticker).history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                total_val += price * asset.shares
                report += f"🔹 **{asset.ticker}**: {asset.shares} shares @ {price:.2f}\n"
        except Exception:
            pass
            
    report += f"\n**Estimated Total Value:** {total_val:,.2f}"
    return {"messages": state["messages"] + [AIMessage(content=report)]}

# ==========================================
# 4. GRAPH COMPILATION
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
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="FinAntinomAI", page_icon="📈")
st.title("💼 FinAntinomAI: Multi-Agent Trading Desk")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("⚠️ API Keys missing! Please add GROQ_API_KEY and TAVILY_API_KEY to your Streamlit Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about a stock, upload a portfolio, or request a fundamental analysis..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Agents are analyzing..."):
            initial_state = {"messages": [HumanMessage(content=prompt)], "next_destination": "", "user_input_type": ""}
            config = {"configurable": {"thread_id": "web_session_1"}}
            
            final_state = app.invoke(initial_state, config=config)
            
            # Handle the Human-In-The-Loop Pause
            if "fundamental" in final_state.get("next_destination", "") or len(final_state.get("messages", [])) == 1:
                final_state = app.invoke(None, config=config)
                
            final_response = final_state['messages'][-1].content
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})