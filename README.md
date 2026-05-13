# rag-and-dine

> AI-powered restaurant recommendation engine built with RAG, multimodal AI, and coordinated LLM agents.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![IBM watsonx](https://img.shields.io/badge/IBM-watsonx.ai-052FAD?logo=ibm&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--db-orange)
![FastMCP](https://img.shields.io/badge/FastMCP-MCP--server-blueviolet)
![Gradio](https://img.shields.io/badge/Gradio-demo-FF7C00?logo=gradio&logoColor=white)
![Coursera](https://img.shields.io/badge/IBM-Coursera%20Capstone-0056D2?logo=coursera&logoColor=white)

---

## What It Does

Transforms a raw California restaurant dataset into a fully interactive recommendation system. Given a user's dining preferences, dietary restrictions, or vibe description, the system retrieves relevant restaurants and recipes using multimodal vector search and synthesizes personalized recommendations through a six-agent AI pipeline.

```
Input : "I want something moody and romantic in DTLA, spicy food preferred"

Output: 1. Iron & Embers — Arts District, DTLA
           Moody industrial steakhouse. 45-day dry-aged ribeye, bone marrow chimichurri.
           Rating: 4.8 | $$$

        2. Marisol Fuego — Downtown LA
           Dim candlelit Mexican kitchen known for fiery mole and smoky mezcal pairings.
           Rating: 4.5 | $$
```

---

## Modules

### Module 1 — Data Preparation

| Lesson | Script | Description |
|--------|--------|-------------|
| 1 | `structure_restaurant_data.py` | Extracts structured JSON from raw text using IBM Granite with one-shot prompting and Pydantic validation |
| 2 | `process_multimodal_data.py` | Generates image captions for food photos and user reviews using LLaMA 4 vision model |
| 3 | `restaurant_data_management.py` | CLI tool for CRUD operations with backup safety and LLM-powered data entry |

### Module 2 — Multimodal RAG System

| Lesson | Script | Description |
|--------|--------|-------------|
| 1 | `construct_multimodal_vector_index.py` | Builds ChromaDB vector indexes using MiniLM text embeddings (384-d) and CLIP image embeddings (512-d) |
| 2 | `similarity_retrieval.py` | Text and image-to-image similarity retrieval with metadata filtering |
| 3 | `multimodal_fusion_ranking.py` | Weighted late fusion ranking across text and image modalities with min-max normalization |

### Module 3 — Multi-Agent Recommendation System

| Lesson | Script | Description |
|--------|--------|-------------|
| 1 | `design_specialized_agents.py` | Defines six specialized agents: User Profile, RAG Retriever, Trend Analyst, Style Expert, Nutrition Expert, Recommendation Expert |
| 2 | `implement_multi_agent_system.py` | Hybrid workflow with sequential and parallel phases using ThreadPoolExecutor |
| 3 | `chatbot_interface.py` | Gradio chatbot with intent classification, preference extraction, and agent pipeline |

### Module 4 — MCP Server and Client

| Lesson | Script | Description |
|--------|--------|-------------|
| 1 | `server.py` + `test.py` | FastMCP server exposing `get_restaurant_info`, `recommend_by_vibe`, and `get_review` as tools |
| 2 | `client.py` | Async MCP client with roots declaration and Claude sampling callback |
| 3 | `app.py` | Gradio chat app with WatsonX LLM running a ReAct agent loop over MCP tools |

---

## Project Structure

```
rag-and-dine/
├── data/
│   ├── California-Culinary-Map.txt
│   ├── Synthetic-User-Reviews.json
│   ├── structured_restaurant_data.json
│   ├── augmented_food_recipe.json
│   └── augmented_user_review.json
├── module1/
├── module2_rag/
├── module3_agents/
├── module4_mcp/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

> This project is designed to run on the IBM Skills Network lab environment. To run locally, uncomment and set your `api_key` in the `Credentials` calls in each script.

## Running

```bash
# Module 1
python module1/structure_restaurant_data.py
python module1/process_multimodal_data.py
python module1/restaurant_data_management.py

# Module 2
python module2_rag/construct_multimodal_vector_index.py
python module2_rag/similarity_retrieval.py
python module2_rag/multimodal_fusion_ranking.py

# Module 3
python module3_agents/design_specialized_agents.py
python module3_agents/implement_multi_agent_system.py
python module3_agents/chatbot_interface.py

# Module 4
python module4_mcp/server.py
python module4_mcp/client.py
python module4_mcp/app.py
```
