# module3_agents/chatbot_interface.py
# Module 3 Lesson 3: Build a Chatbot Interface for the Recommendation System
# Gradio-based chatbot that classifies user intent, extracts preferences,
# runs the multi-agent workflow, and presents formatted recommendations.

# ── Imports ────────────────────────────────────────────────────────────────────

import os
import json
import gradio as gr
from typing import List, Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Configuration ──────────────────────────────────────────────────────────────

# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── Step 1: Intent Classification ─────────────────────────────────────────────

def classify_intent(user_message: str, llm: ChatOpenAI) -> str:
    """Classify user intent as restaurant, recipe, both, clarification, or database."""
    system_prompt = """You are an intent classifier for a food recommendation system.

Analyze the user's message and classify it as ONE of:
- "restaurant" - User wants restaurant recommendations
- "recipe"     - User wants recipe recommendations
- "both"       - User wants both restaurant and recipe recommendations
- "clarification" - User needs help or is asking a question
- "database"   - User wants to add/edit/delete database entries

Examples:
"Where should I eat tonight?"  → restaurant
"How do I make lasagna?"       → recipe
"I want dinner ideas"          → both
"What can you help me with?"   → clarification
"I want to add a new restaurant" → database

Respond with ONLY the classification label."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    response = llm.invoke(messages)
    intent   = response.content.strip().lower()

    valid_intents = ["restaurant", "recipe", "both", "clarification", "database"]
    return intent if intent in valid_intents else "clarification"

# ── Step 2: Preference Extraction ──────────────────────────────────────────────

def extract_preferences(user_message: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """Extract structured preferences from natural language input."""
    system_prompt = """You are a preference extractor for a food recommendation system.

Extract user preferences from their message and return JSON with these keys:
- favorite_cuisines:    List of mentioned cuisines (e.g., ["Italian", "Thai"])
- dietary_restrictions: List of dietary needs (e.g., ["vegetarian", "gluten-free"])
- dining_occasion:      Type of dining (e.g., "casual", "fine dining", "quick bite")
- price_range:          Price preference (e.g., "$", "$$", "$$$", "$$$$")
- flavor_preferences:   List of flavor preferences (e.g., ["spicy", "sweet"])
- other_preferences:    Any other relevant details

If a field is not mentioned, use an empty list or "not specified".

Example:
Input: "I love spicy Thai food and I'm vegetarian"
Output: {
  "favorite_cuisines": ["Thai"],
  "dietary_restrictions": ["vegetarian"],
  "dining_occasion": "not specified",
  "price_range": "not specified",
  "flavor_preferences": ["spicy"],
  "other_preferences": ""
}

Respond with ONLY valid JSON."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    response = llm.invoke(messages)

    try:
        return json.loads(response.content)
    except Exception:
        return {
            "favorite_cuisines":    [],
            "dietary_restrictions": [],
            "dining_occasion":      "not specified",
            "price_range":          "not specified",
            "flavor_preferences":   [],
            "other_preferences":    ""
        }

# ── Step 3: Multi-Agent Workflow ───────────────────────────────────────────────

def run_recommendation_workflow(preferences: Dict[str, Any], recommendation_type: str) -> Dict[str, Any]:
    """
    Run the multi-agent workflow and return recommendations.
    In production this calls the LangGraph workflow from Lesson 2.
    Here we return mock recommendations for demonstration.
    """
    print(f"Running workflow for {recommendation_type} recommendations...")

    mock_recommendations = {
        "restaurants": [
            {
                "name":      "Green Leaf Bistro",
                "cuisine":   "Mediterranean",
                "price":     "$$",
                "reasoning": "This restaurant perfectly aligns with your preference for healthy, plant-based options and offers a diverse Mediterranean menu with excellent vegetarian choices."
            },
            {
                "name":      "Spice Route",
                "cuisine":   "Indian",
                "price":     "$$",
                "reasoning": "Known for authentic Indian cuisine with extensive vegetarian options. The spice level can be customized to your preference."
            }
        ],
        "recipes": [
            {
                "name":       "One-Pot Chickpea Curry",
                "cuisine":    "Indian",
                "difficulty": "Easy",
                "reasoning":  "A flavorful, protein-rich dish that matches your love for bold flavors. Ready in 30 minutes with simple ingredients."
            },
            {
                "name":       "Mediterranean Quinoa Bowl",
                "cuisine":    "Mediterranean",
                "difficulty": "Easy",
                "reasoning":  "Nutritious and satisfying, this bowl combines your favorite Mediterranean flavors with plant-based protein."
            }
        ]
    }

    if recommendation_type == "restaurant":
        return {"restaurants": mock_recommendations["restaurants"]}
    elif recommendation_type == "recipe":
        return {"recipes": mock_recommendations["recipes"]}
    else:
        return mock_recommendations

# ── Step 4: Format Recommendations ────────────────────────────────────────────

def format_recommendations(recommendations: Dict[str, Any]) -> str:
    """Format recommendations for display in the chat interface."""
    output = ""

    if recommendations.get("restaurants"):
        output += "🍽️ **Restaurant Recommendations:**\n\n"
        for i, r in enumerate(recommendations["restaurants"], 1):
            output += f"**{i}. {r['name']}**\n"
            output += f"   - Cuisine: {r['cuisine']}\n"
            output += f"   - Price: {r['price']}\n"
            output += f"   - Why: {r['reasoning']}\n\n"

    if recommendations.get("recipes"):
        output += "👨‍🍳 **Recipe Recommendations:**\n\n"
        for i, r in enumerate(recommendations["recipes"], 1):
            output += f"**{i}. {r['name']}**\n"
            output += f"   - Cuisine: {r['cuisine']}\n"
            output += f"   - Difficulty: {r['difficulty']}\n"
            output += f"   - Why: {r['reasoning']}\n\n"

    return output or "I couldn't generate recommendations. Please try again with more details about your preferences."

# ── Step 5: Main Chatbot Function ──────────────────────────────────────────────

def recommendation_chatbot(message: str, history: List[Tuple[str, str]]) -> str:
    """Main chatbot handler — classifies intent, extracts preferences, returns recommendations."""
    try:
        intent = classify_intent(message, llm)
        print(f"Classified intent: {intent}")

        if intent == "clarification":
            return """I'm your food recommendation assistant! I can help you with:

🍽️ **Restaurant recommendations** - Tell me your cuisine preferences, dietary restrictions, and occasion
👨‍🍳 **Recipe recommendations** - Let me know what you'd like to cook
📝 **Database management** - Add, update, or delete restaurants and recipes

Just describe what you're looking for and I'll provide personalized recommendations!"""

        elif intent == "database":
            return """To manage the database, please use the tabs above:

- **Add Restaurant**: Submit a new restaurant
- **Add Recipe**: Submit a new recipe
- **Edit/Delete**: Modify or remove existing entries

Is there anything else I can help you with?"""

        elif intent in ["restaurant", "recipe", "both"]:
            preferences     = extract_preferences(message, llm)
            print(f"Extracted preferences: {preferences}")
            recommendations = run_recommendation_workflow(preferences, intent)
            return format_recommendations(recommendations)

        else:
            return "I'm not sure how to help with that. Can you rephrase your request?"

    except Exception as e:
        return f"I encountered an error: {str(e)}. Please make sure you have set your OpenAI API key."

# ── Step 6: Database Management Functions ─────────────────────────────────────

def add_restaurant(name: str, cuisine: str, price: str, location: str, description: str) -> str:
    """Add a new restaurant to the database."""
    print(f"Adding restaurant: {name}")
    return f"✅ Successfully added '{name}' to the database!"

def add_recipe(name: str, cuisine: str, difficulty: str, prep_time: str, ingredients: str, instructions: str) -> str:
    """Add a new recipe to the database."""
    print(f"Adding recipe: {name}")
    return f"✅ Successfully added '{name}' recipe to the database!"

# ── Step 7: Gradio Interface ───────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    """Build the complete Gradio interface with four tabs."""
    with gr.Blocks(title="Food Recommendation Chatbot", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🍽️ Food Recommendation Chatbot
        Your personal AI assistant for restaurant and recipe recommendations!
        """)

        with gr.Tabs():

            # Tab 1: Chat
            with gr.Tab("💬 Chat"):
                gr.ChatInterface(
                    fn=recommendation_chatbot,
                    examples=[
                        "I'm looking for vegetarian restaurants",
                        "Suggest some easy recipes for dinner",
                        "I want spicy Thai food recommendations",
                        "What can you help me with?"
                    ],
                    title="Chat with the Recommendation Assistant",
                    description="Describe your food preferences and I'll recommend restaurants or recipes!"
                )

            # Tab 2: Add Restaurant
            with gr.Tab("➕ Add Restaurant"):
                gr.Markdown("### Add a New Restaurant to the Database")
                with gr.Row():
                    with gr.Column():
                        rest_name        = gr.Textbox(label="Restaurant Name")
                        rest_cuisine     = gr.Textbox(label="Cuisine Type")
                        rest_price       = gr.Dropdown(choices=["$", "$$", "$$$", "$$$$"], label="Price Range")
                    with gr.Column():
                        rest_location    = gr.Textbox(label="Location")
                        rest_description = gr.Textbox(label="Description", lines=3)

                add_rest_btn = gr.Button("Add Restaurant", variant="primary")
                rest_output  = gr.Textbox(label="Status")
                add_rest_btn.click(
                    fn=add_restaurant,
                    inputs=[rest_name, rest_cuisine, rest_price, rest_location, rest_description],
                    outputs=rest_output
                )

            # Tab 3: Add Recipe
            with gr.Tab("➕ Add Recipe"):
                gr.Markdown("### Add a New Recipe to the Database")
                with gr.Row():
                    with gr.Column():
                        recipe_name       = gr.Textbox(label="Recipe Name")
                        recipe_cuisine    = gr.Textbox(label="Cuisine Type")
                        recipe_difficulty = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="Difficulty")
                    with gr.Column():
                        recipe_time        = gr.Textbox(label="Prep Time")
                        recipe_ingredients = gr.Textbox(label="Ingredients (comma-separated)", lines=3)

                recipe_instructions = gr.Textbox(label="Instructions", lines=5)
                add_recipe_btn      = gr.Button("Add Recipe", variant="primary")
                recipe_output       = gr.Textbox(label="Status")
                add_recipe_btn.click(
                    fn=add_recipe,
                    inputs=[recipe_name, recipe_cuisine, recipe_difficulty, recipe_time, recipe_ingredients, recipe_instructions],
                    outputs=recipe_output
                )

            # Tab 4: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This Chatbot

                This chatbot uses a multi-agent AI system to provide personalized food recommendations.

                ### Features:
                - 🤖 **Intelligent Agents**: Six specialized AI agents work together
                - 🔍 **Smart Search**: Vector database retrieval finds the most relevant options
                - 🎯 **Personalized**: Recommendations tailored to your tastes and dietary needs
                - 📝 **Editable Database**: Add your favorite restaurants and recipes

                ### How to Use:
                1. Go to the **Chat** tab
                2. Describe what you're looking for
                3. Receive personalized recommendations
                4. Use the **Add** tabs to contribute to the database

                ### Technologies:
                - LangChain & LangGraph for multi-agent orchestration
                - OpenAI GPT-4 for language understanding
                - Vector databases for semantic search
                - Gradio for the user interface
                """)

    return demo

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Module 3 Lesson 3: Chatbot Interface")
    print("=" * 60)

    # Task 1: Test preference extraction
    print("\n--- Task 1: Preference Extraction Test ---")
    test_message = "I'm looking for affordable vegetarian Mexican restaurants"
    print(f"Input: {test_message}")
    try:
        preferences = extract_preferences(test_message, llm)
        print("Extracted Preferences:")
        print(json.dumps(preferences, indent=2))
    except Exception as e:
        print(f"Requires valid OpenAI API key. Error: {e}")

    # Task 2: Test chatbot with recipe request
    print("\n--- Task 2: Recipe Request Test ---")
    test_recipe_message = "I want to cook a spicy Thai curry at home. Any easy recipe suggestions?"
    print(f"User: {test_recipe_message}\n")
    try:
        response = recommendation_chatbot(test_recipe_message, [])
        print("Bot Response:")
        print(response)
    except Exception as e:
        print(f"Requires valid OpenAI API key. Error: {e}")

    # Launch Gradio interface
    print("\n--- Launching Gradio Interface ---")
    demo = build_interface()
    demo.launch(share=True)


if __name__ == "__main__":
    main()
