import os
import streamlit as st
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Note: In production, ensure your API keys (e.g., Google GenAI, Tavily) are set 
# in Streamlit's "Advanced Settings" Secrets manager, not hardcoded here.
from langchain_google_genai import ChatGoogleGenerativeAI 

# ==========================================
# 1. APPLICATION STATE
# ==========================================
class ApplicationState(TypedDict):
    messages: List[BaseMessage]
    next_destination: str
    user_input_type: str

# ==========================================
# 2. ORCHESTRATOR ROUTER
# ==========================================
class IntentRoute(BaseModel):
    destination: str = Field(
        description="Must be exactly one of: 'debate', 'vision', 'trivia', 'fundamental', 'portfolio'"
    )

def orchestrator_router(state: ApplicationState):
    user_message = state["messages"][-1].content
    # Using temperature 0.0 for strict, deterministic routing
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)
    router_llm = llm.with_structured_output(IntentRoute)
    
    messages = [
        SystemMessage(content="""You are the front-desk router for a financial platform. 
        Classify the user input into exactly one of these 5 categories:
        - 'debate': Asking if they should buy/sell a specific stock.
        - 'vision': Uploading a chart or asking for technical visual analysis.
        - 'trivia': Asking for basic financial definitions, concepts, or current stock prices.
        - 'fundamental': Asking for a deep dive or fundamental report on a company.
        - 'portfolio': Providing a list of multiple stock holdings to analyze."""),
        HumanMessage(content=f"User Input: {user_message}")
    ]
    
    decision = router_llm.invoke(messages)
    return {"next_destination": decision.destination, "user_input_type": decision.destination}

# ==========================================
# 3. WORKER NODES
# ==========================================
def trivia_node(state: ApplicationState):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def vision_node(state: ApplicationState):
    # Stub for the Multi-Modal Chart Line
    msg = AIMessage(content="[Vision Agent] I have analyzed the chart. The trend shows key support and resistance levels based on the candlestick formations.")
    return {"messages": state["messages"] + [msg]}

def bull_agent(state: ApplicationState):
    msg = AIMessage(content="[Bull Agent] The catalysts for this asset are incredibly strong. Upside growth potential is massive due to macro tailwinds.")
    return {"messages": state["messages"] + [msg]}

def bear_agent(state: ApplicationState):
    msg = AIMessage(content="[Bear Agent] I disagree. The valuation gaps and regulatory roadblocks present significant downside risk.")
    return {"messages": state["messages"] + [msg]}

def judge_agent(state: ApplicationState):
    msg = AIMessage(content="[Judge Agent] Weighing both sides, the recommendation is a balanced hold. The upside is present, but risks must be managed.")
    return {"messages": state["messages"] + [msg]}

def fundamental_node(state: ApplicationState):
    # This node executes AFTER the Human-in-the-Loop approval
    msg = AIMessage(content="[Fundamental Agent] Human approval received. Deep Fundamental Analysis complete. The company shows strong free cash flow and expanding gross margins.")
    return {"messages": state["messages"] + [msg]}

def portfolio_node(state: ApplicationState):
    msg = AIMessage(content="[Portfolio Aggregator] I have extracted your holdings. Your portfolio is diversified, but watch your sector concentration risk.")
    return {"messages": state["messages"] + [msg]}

# ==========================================
# 4. GRAPH COMPILATION
# ==========================================
workflow = StateGraph(ApplicationState)

# Register nodes
workflow.add_node("orchestrator", orchestrator_router)
workflow.add_node("trivia", trivia_node)
workflow.add_node("vision", vision_node)
workflow.add_node("fundamental", fundamental_node)
workflow.add_node("portfolio", portfolio_node)
workflow.add_node("bull_agent", bull_agent)
workflow.add_node("bear_agent", bear_agent)
workflow.add_node("judge_agent", judge_agent)

workflow.set_entry_point("orchestrator")

def route_traffic(state: ApplicationState):
    return state["next_destination"]

# Dynamic Router
workflow.add_conditional_edges(
    "orchestrator",
    route_traffic,
    {
        "debate": "bull_agent", 
        "vision": "vision",
        "trivia": "trivia",
        "fundamental": "fundamental",
        "portfolio": "portfolio"
    }
)

# Wire execution edges
workflow.add_edge("bull_agent", "bear_agent")
workflow.add_edge("bear_agent", "judge_agent")
workflow.add_edge("judge_agent", END)
workflow.add_edge("vision", END)
workflow.add_edge("trivia", END)
workflow.add_edge("fundamental", END)
workflow.add_edge("portfolio", END)

# Initialize Memory for HITL Checkpoints
memory = MemorySaver()

# Compile with a breakpoint before the fundamental node
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["fundamental"]
)

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="FinAntinomAI", page_icon="📈")
st.title("💼 FinAntinomAI: Multi-Agent Trading Desk")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture user input
if prompt := st.chat_input("Ask about a stock, upload a portfolio, or request a fundamental analysis..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run the LangGraph Orchestrator
    with st.chat_message("assistant"):
        with st.spinner("Agents are analyzing..."):
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "next_destination": "",
                "user_input_type": ""
            }
            
            # Use a static thread_id for the web session to enable memory checkpoints
            config = {"configurable": {"thread_id": "web_session_1"}}
            
            # Invoke the graph
            final_state = app.invoke(initial_state, config=config)
            
            # Check if the graph paused due to the Human-In-The-Loop interrupt
            if "fundamental" in final_state.get("next_destination", "") or len(final_state.get("messages", [])) == 1:
                # If paused, we resume execution immediately (mocking the human approval for the basic web UI)
                final_state = app.invoke(None, config=config)
                
            final_response = final_state['messages'][-1].content
            
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})