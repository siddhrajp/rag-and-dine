# module4_mcp/app.py
# Module 4 Lesson 3: Build a Full MCP Application
# A Gradio chat interface backed by WatsonX LLM that uses a ReAct agent loop
# to discover and call MCP tools intelligently.

# ── Imports and Configuration ──────────────────────────────────────────────────

import os
import gradio as gr
from pathlib import Path
from fastmcp.client import Client, PythonStdioTransport
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")

SYSTEM_PROMPT = """You are Connoisseur Companion, a knowledgeable and friendly AI guide
to California's restaurant scene.

You have access to a database of California restaurants and can help users discover
great places to eat based on their preferences.

You have access to the following tools:
- get_restaurant_info: Use this to look up specific restaurants by name. Call this when
  a user asks about a particular restaurant or wants details like cuisine, rating,
  price range, or signature dishes.
- recommend_by_vibe: Use this to find restaurants that match a mood or atmosphere.
  Call this when a user describes a feeling, ambiance, or vibe they are looking for
  (e.g., "moody", "romantic", "zen", "sun-drenched", "lively").
- get_review: Use this to retrieve detailed reviews of restaurants. Call this when
  a user wants to know what others think about a specific restaurant.

Always use the appropriate tool to find accurate information before responding.
Be warm, enthusiastic, and helpful — like a knowledgeable friend who knows all the
best spots in California."""

project_id = (
    os.environ.get("WATSONX_AI_PROJECT_ID")
    or os.environ.get("WATSONX_PROJECT_ID")
)

# ── WatsonX LLM Factory ────────────────────────────────────────────────────────

def make_model():
    """Create a fresh WatsonX LLM instance for each conversation turn."""
    return ChatWatsonx(
        model_id="ibm/granite-4-h-small",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params={"temperature": 0.7},
    )

# ── ReAct Agent Loop ───────────────────────────────────────────────────────────

async def chat_with_agent(user_message: str, history: list) -> str:
    """Connect to the MCP server, discover tools, and run a ReAct loop.
    The LLM decides which tools to call, calls them via the MCP server,
    and repeats until it produces a final text response."""

    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:

        # Step 1: Discover available tools from the MCP server at runtime
        mcp_tools = await client.list_tools()

        # Step 2: Convert MCP tool schemas to OpenAI-style tool definitions for the LLM
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]

        # Bind tools to the LLM so it knows what it can call
        model = make_model().bind_tools(openai_tools)

        # Step 3: Build message list from chat history and new user message
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        # Step 4: ReAct loop — call tools until the LLM returns a plain text reply
        for _ in range(10):
            response = await model.ainvoke(messages)
            messages.append(response)

            # No tool calls means the LLM is done — return the final response
            if not response.tool_calls:
                raw = response.content
                if isinstance(raw, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                return str(raw)

            # Execute each tool call via the MCP server and feed results back
            for tool_call in response.tool_calls:
                result = await client.call_tool(tool_call["name"], tool_call["args"])
                tool_output = " ".join(
                    item.text if hasattr(item, "text") else str(item)
                    for item in result.content
                ) if result.content else "(no result)"
                messages.append(
                    ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
                )

    return "I wasn't able to complete that request. Please try again."

# ── Gradio Event Handler ───────────────────────────────────────────────────────

async def handle_chat(user_message, history):
    """Wrap chat_with_agent with optimistic UI update showing Thinking... placeholder."""
    if history is None:
        history = []

    if not user_message or not user_message.strip():
        yield history
        return

    # Show the user's message and a Thinking... placeholder immediately
    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield history

    # Run the agent and replace the placeholder with the real answer
    response_text = await chat_with_agent(user_message, history[:-2])
    history[-1]   = {"role": "assistant", "content": response_text}
    yield history

# ── Gradio Interface ───────────────────────────────────────────────────────────

with gr.Blocks(title="Connoisseur Companion", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# Connoisseur Companion\n"
        "Your AI guide to California's restaurant scene. "
        "Ask me about restaurants by name, cuisine, or vibe!"
    )

    chatbot   = gr.Chatbot(height=500)
    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder=(
            'e.g., "Find me a moody spot in DTLA" or '
            '"Tell me about Sakura Garden"'
        ),
    )

    with gr.Row():
        btn1 = gr.Button("Find moody restaurants",       size="sm")
        btn2 = gr.Button("Tell me about Iron & Embers",  size="sm")
        btn3 = gr.Button("Zen dining in Little Tokyo?",  size="sm")

    # Submit on Enter — also clear the text box
    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot])
    msg_input.submit(lambda: "", None, msg_input)

    # Quick-start buttons inject pre-written prompts via gr.State
    btn1.click(
        handle_chat,
        [gr.State("Find me some moody restaurants"), chatbot],
        [chatbot]
    )
    btn2.click(
        handle_chat,
        [gr.State("Tell me about Iron & Embers"), chatbot],
        [chatbot]
    )
    btn3.click(
        handle_chat,
        [gr.State("What's a zen dining experience in Little Tokyo?"), chatbot],
        [chatbot]
    )

# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Connoisseur Companion...")
    demo.launch(
        share=True,
        theme=gr.themes.Soft(),
    )
