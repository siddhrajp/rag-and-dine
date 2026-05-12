# rag-and-dine

An AI-powered restaurant recommendation system built as part of the IBM Coursera capstone course **"RAG and Agentic AI Capstone Project"**.

The project transforms messy, unstructured restaurant data into an intelligent recommendation engine using RAG (Retrieval Augmented Generation), multimodal AI, and coordinated LLM agents.

## Tech Stack

- **IBM watsonx.ai** — Granite LLM (`ibm/granite-4-h-small`) and LLaMA 4 vision model
- **LangChain & LangGraph** — RAG pipeline and multi-agent orchestration
- **Pydantic** — JSON schema validation
- **Python**

## Project Structure

```
rag-and-dine/
├── data/                            # Raw inputs and processed outputs
│   ├── California-Culinary-Map.txt
│   ├── Synthetic-User-Reviews.json
│   ├── structured_restaurant_data.json
│   ├── augmented_food_recipe.json
│   └── augmented_user_review.json
├── module1/                         # Module 1: Data Preparation
│   ├── structure_restaurant_data.py
│   ├── process_multimodal_data.py
│   └── restaurant_data_management.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Modules

### Module 1 — Data Preparation

| Lesson | Script | Description |
|--------|--------|-------------|
| Lesson 1 | `structure_restaurant_data.py` | Extracts structured JSON from raw restaurant text using IBM Granite LLM with one-shot prompting and Pydantic validation |
| Lesson 2 | `process_multimodal_data.py` | Generates image captions for food recipe photos and user review images using LLaMA 4 vision model |
| Lesson 3 | `restaurant_data_management.py` | CLI tool for browsing, adding, editing, and deleting restaurant records with backup safety and LLM-powered data entry |

### Module 2 — Multimodal RAG System *(coming soon)*

### Module 3 — Multi-Agent Recommendation System *(coming soon)*

### Module 4 — RAG Pipeline with LangChain *(coming soon)*

### Module 5 — Final Integrated Application *(coming soon)*

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** This project is designed to run on the IBM Skills Network lab environment. To run locally, add your `api_key` to the `Credentials` calls in each script.

## Running

```bash
# Lesson 1 — Structure raw restaurant text into JSON
python module1/structure_restaurant_data.py

# Lesson 2 — Generate image captions
python module1/process_multimodal_data.py

# Lesson 3 — Launch the CLI data management UI
python module1/restaurant_data_management.py
```
