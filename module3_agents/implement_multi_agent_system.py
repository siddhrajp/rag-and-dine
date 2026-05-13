# module3_agents/implement_multi_agent_system.py
# Module 3 Lesson 2: Implement and Test a Multi-Agent Recommendation System
# Integrates six specialized agents into a hybrid workflow using ThreadPoolExecutor
# for parallel execution of analysis agents.

# ── Imports ────────────────────────────────────────────────────────────────────

import os
import json
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
client = OpenAI()
MODEL  = "gpt-4o"

# ── Agent Configurations ───────────────────────────────────────────────────────

agent_configs = {
    "user_profile_generator": {
        "role":      "User Profile Generator",
        "goal":      "Analyze user restaurant visit history and social media posts to create a comprehensive profile.",
        "backstory": "You are an expert user behavior analyst with 10 years of experience in the food industry. You excel at identifying patterns in dining behavior and building rich user profiles."
    },
    "rag_retriever": {
        "role":      "RAG Retriever",
        "goal":      "Query multimodal vector databases to retrieve relevant restaurants and recipes.",
        "backstory": "You are a data retrieval specialist with expertise in vector databases and semantic search."
    },
    "food_trend_analyst": {
        "role":      "Food Trend Analyst",
        "goal":      "Identify current food trends and emerging dining concepts.",
        "backstory": "You are a culinary journalist who has spent 15 years covering food culture across global markets."
    },
    "food_style_expert": {
        "role":      "Food Style Expert",
        "goal":      "Analyze cuisine types and flavor profiles to match user preferences.",
        "backstory": "You are a trained chef and culinary anthropologist with expertise in global cuisines."
    },
    "nutrition_expert": {
        "role":      "Nutrition Expert",
        "goal":      "Evaluate nutritional content and ensure dietary compliance.",
        "backstory": "You are a registered dietitian with 8 years of clinical experience."
    },
    "recommendation_expert": {
        "role":      "Recommendation Expert",
        "goal":      "Synthesize insights from all agents into final recommendations.",
        "backstory": "You are a recommendation systems architect with experience in personalization engines."
    }
}

# ── Shared State Template ──────────────────────────────────────────────────────

INITIAL_STATE = {
    "user_input":             "",    # Phase 1 input
    "user_profile":           {},    # Phase 1 output
    "retrieved_restaurants":  [],    # Phase 2 output
    "retrieved_recipes":      [],    # Phase 2 output
    "trend_analysis":         {},    # Phase 3a output
    "style_analysis":         {},    # Phase 3b output
    "nutrition_analysis":     {},    # Phase 3c output
    "final_recommendations":  {},    # Phase 4 output
    "workflow_step":          "start"
}

# ── Helper: Call Agent ─────────────────────────────────────────────────────────

def call_agent(agent_key: str, user_message: str) -> str:
    """Build system prompt from config and call the LLM."""
    config = agent_configs[agent_key]
    system_prompt = f"""You are a {config['role']}.

Your goal: {config['goal']}

Your background: {config['backstory']}

Respond with structured, actionable output."""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]
    )
    return response.choices[0].message.content

# ── Phase 1: User Profile (Sequential) ────────────────────────────────────────

def node_generate_profile(state: dict) -> dict:
    """Phase 1 — Generate user profile from input data."""
    print("\n[Phase 1] Generating user profile...")

    user_message = f"""Analyze this user data and create a comprehensive profile:

{state['user_input']}

Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- summary (string)
"""
    try:
        response     = call_agent("user_profile_generator", user_message)
        user_profile = json.loads(response)
        print(f"✓ User profile generated: {user_profile.get('summary', 'No summary')}")
    except Exception as e:
        print(f"⚠ Error generating profile: {e}")
        user_profile = {"error": str(e)}

    state["user_profile"]  = user_profile
    state["workflow_step"] = "profile_generated"
    return state

# ── Phase 2: Retrieve Candidates (Sequential) ──────────────────────────────────

def node_retrieve_candidates(state: dict) -> dict:
    """Phase 2 — Retrieve restaurant and recipe candidates."""
    print("\n[Phase 2] Retrieving candidates from vector database...")

    profile = state["user_profile"]
    user_message = f"""Based on this user profile:
{json.dumps(profile, indent=2)}

Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.

Return JSON with two arrays:
- restaurants: [{{"name": str, "cuisine": str, "price": str, "rating": float, "description": str}}]
- recipes:     [{{"name": str, "cuisine": str, "difficulty": str, "prep_time": str, "description": str}}]

Make the results realistic and diverse.
"""
    try:
        response       = call_agent("rag_retriever", user_message)
        retrieved_data = json.loads(response)
        restaurants    = retrieved_data.get("restaurants", [])
        recipes        = retrieved_data.get("recipes", [])
        print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    except Exception as e:
        print(f"⚠ Error retrieving candidates: {e}")
        restaurants, recipes = [], []

    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"]     = recipes
    state["workflow_step"]         = "candidates_retrieved"
    return state

# ── Phase 3a: Trend Analysis (Parallel) ───────────────────────────────────────

def node_analyze_trends(state: dict) -> dict:
    """Phase 3a — Analyze food trends in retrieved candidates."""
    print("\n[Phase 3a] Analyzing food trends...")

    restaurants = state["retrieved_restaurants"]
    recipes     = state["retrieved_recipes"]

    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes:     {json.dumps(recipes[:5], indent=2)}

Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    try:
        response       = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"✓ Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"⚠ Error analyzing trends: {e}")
        trend_analysis = {"error": str(e)}

    state["trend_analysis"] = trend_analysis
    return state

# ── Phase 3b: Style Analysis (Parallel) ───────────────────────────────────────

def node_analyze_styles(state: dict) -> dict:
    """Phase 3b — Analyze cuisine types and flavor profiles."""
    print("\n[Phase 3b] Analyzing food styles...")

    restaurants = state["retrieved_restaurants"]
    recipes     = state["retrieved_recipes"]
    profile     = state["user_profile"]

    user_message = f"""Analyze the cuisine types, cooking methods, and flavor profiles
of these retrieved options:

User Profile:  {json.dumps(profile, indent=2)}
Restaurants:   {json.dumps(restaurants[:5], indent=2)}
Recipes:       {json.dumps(recipes[:5], indent=2)}

Identify:
- Cuisine categories and regional variations present
- Key cooking techniques used
- Dominant flavor profiles
- How each item maps to the user's flavor preferences
- Discovery suggestions: cuisines the user hasn't tried but would enjoy

Return JSON:
{{
  "cuisine_breakdown":      [{{"cuisine": str, "count": int, "description": str}}],
  "flavor_profile_mapping": [{{"item": str, "flavor_profile": str, "match_score": str}}],
  "top_picks":              [{{"name": str, "reason": str}}],
  "discovery_suggestions":  [{{"cuisine": str, "why": str}}]
}}
"""
    try:
        response       = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print("✓ Style analysis completed")
    except Exception as e:
        print(f"⚠ Error analyzing styles: {e}")
        style_analysis = {"error": str(e)}

    state["style_analysis"] = style_analysis
    return state

# ── Phase 3c: Nutrition Evaluation (Parallel) ──────────────────────────────────

def node_evaluate_nutrition(state: dict) -> dict:
    """Phase 3c — Evaluate nutritional aspects and dietary compliance."""
    print("\n[Phase 3c] Evaluating nutrition...")

    restaurants = state["retrieved_restaurants"]
    recipes     = state["retrieved_recipes"]
    profile     = state["user_profile"]

    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants:  {json.dumps(restaurants[:5], indent=2)}
Recipes:      {json.dumps(recipes[:5], indent=2)}

Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    try:
        response           = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print("✓ Nutrition evaluation completed")
    except Exception as e:
        print(f"⚠ Error evaluating nutrition: {e}")
        nutrition_analysis = {"error": str(e)}

    state["nutrition_analysis"] = nutrition_analysis
    return state

# ── Phase 4: Generate Recommendations (Sequential) ────────────────────────────

def node_generate_recommendations(state: dict) -> dict:
    """Phase 4 — Synthesize all analyses into final recommendations."""
    print("\n[Phase 4] Generating final recommendations...")

    user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants:  {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
Recipes:      {json.dumps(state['retrieved_recipes'][:10], indent=2)}
Trends:       {json.dumps(state['trend_analysis'], indent=2)}
Styles:       {json.dumps(state['style_analysis'], indent=2)}
Nutrition:    {json.dumps(state['nutrition_analysis'], indent=2)}

Return JSON:
{{
  "restaurants": [{{"name": str, "reasoning": str}}],
  "recipes":     [{{"name": str, "reasoning": str}}]
}}

Each reasoning should be 2-3 sentences explaining why it's a great match.
"""
    try:
        response        = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"✓ Generated {len(recommendations.get('restaurants', []))} restaurant recommendations")
        print(f"✓ Generated {len(recommendations.get('recipes', []))} recipe recommendations")
    except Exception as e:
        print(f"⚠ Error generating recommendations: {e}")
        recommendations = {"error": str(e)}

    state["final_recommendations"] = recommendations
    state["workflow_step"]         = "complete"
    return state

# ── Hybrid Workflow Orchestrator ───────────────────────────────────────────────

def run_workflow(user_input: str) -> dict:
    """
    Run the full hybrid multi-agent workflow.
    Phase 1: User Analysis      (sequential)
    Phase 2: Data Retrieval     (sequential)
    Phase 3: Analysis           (parallel — trends, styles, nutrition)
    Phase 4: Synthesis          (sequential)
    """
    state = {**INITIAL_STATE, "user_input": user_input}

    # Phase 1
    state = node_generate_profile(state)

    # Phase 2
    state = node_retrieve_candidates(state)

    # Phase 3 — Parallel
    print("\n[Phase 3] Running analysis agents in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_trends    = executor.submit(node_analyze_trends,     dict(state))
        future_styles    = executor.submit(node_analyze_styles,     dict(state))
        future_nutrition = executor.submit(node_evaluate_nutrition, dict(state))

        result_trends    = future_trends.result()
        result_styles    = future_styles.result()
        result_nutrition = future_nutrition.result()

    state["trend_analysis"]     = result_trends["trend_analysis"]
    state["style_analysis"]     = result_styles["style_analysis"]
    state["nutrition_analysis"] = result_nutrition["nutrition_analysis"]

    # Phase 4
    state = node_generate_recommendations(state)

    return state

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_recommendations(result: Dict[str, Any]):
    """Evaluate the quality of final recommendations."""
    print("\n" + "=" * 60)
    print("RECOMMENDATION EVALUATION")
    print("=" * 60)

    profile         = result.get("user_profile", {})
    recommendations = result.get("final_recommendations", {})
    restaurants     = recommendations.get("restaurants", [])
    recipes         = recommendations.get("recipes", [])

    print(f"\n✓ Restaurant recommendations: {len(restaurants)}")
    print(f"✓ Recipe recommendations:     {len(recipes)}")

    dietary = profile.get("dietary_restrictions", [])
    if dietary:
        print(f"\n✓ Dietary restrictions: {', '.join(dietary)}")

    cuisines = profile.get("favorite_cuisines", [])
    if cuisines:
        print(f"✓ Favourite cuisines: {', '.join(cuisines)}")

    if restaurants:
        print(f"\n✓ Top restaurant: {restaurants[0].get('name', 'N/A')}")
        print(f"  Reasoning: {restaurants[0].get('reasoning', 'N/A')}")

    if recipes:
        print(f"\n✓ Top recipe: {recipes[0].get('name', 'N/A')}")
        print(f"  Reasoning: {recipes[0].get('reasoning', 'N/A')}")

    print("\n" + "=" * 60)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Module 3 Lesson 2: Multi-Agent Recommendation System")
    print("=" * 60)

    # Test Case 1: Health-Conscious User
    test_user_1 = """
Restaurant Visit History:
- Visited "Green Bowl" (Vegan, $$) 8 times
- Visited "Mediterranean Grill" (Mediterranean, $$) 5 times
- Visited "Juice Lab" (Smoothies, $) 3 times

Social Media Posts:
- "Loving my plant-based journey! 🌱"
- "This gluten-free Mediterranean bowl is amazing!"
- "Fresh juice is the best way to start the day."

Dietary Restrictions: Vegan, Gluten-Free
"""

    # Test Case 2: Adventurous Foodie
    test_user_2 = """
Restaurant Visit History:
- Visited "Omakase Sushi" (Japanese Fine Dining, $$$$) 4 times
- Visited "Street Food Market" (International Fusion, $$) 6 times
- Visited "Molecular Gastronomy Lab" (Experimental, $$$$) 2 times

Social Media Posts:
- "Mind-blown by the 12-course tasting menu! 🤯"
- "Trying crickets for the first time. Surprisingly good!"
- "This molecular take on traditional ramen is art."

Dietary Restrictions: None
"""

    for i, (label, user_input) in enumerate([
        ("Health-Conscious User", test_user_1),
        ("Adventurous Foodie",    test_user_2)
    ], start=1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {label}")
        print("=" * 60)
        try:
            result = run_workflow(user_input)
            print("\n" + "=" * 60)
            print("FINAL RECOMMENDATIONS")
            print("=" * 60)
            print(json.dumps(result["final_recommendations"], indent=2))
            evaluate_recommendations(result)
        except Exception as e:
            print(f"\nTest requires a valid OpenAI API key. Error: {e}")


if __name__ == "__main__":
    main()
