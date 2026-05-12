# module1_structure_data/structure_restaurant_data.py
# Module 1: Structure unstructured restaurant text data into JSON using IBM Granite LLM

import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# ── Load Data ──────────────────────────────────────────────────────────────────

file_path = "data/California-Culinary-Map.txt"

with open(file_path, 'r') as file:
    restaurant_data = file.read()

restaurant_list = restaurant_data.split("\n\n")[1:]  # Remove dataset title
print(f"Total restaurants loaded: {len(restaurant_list)}")

# ── LLM Model ──────────────────────────────────────────────────────────────────

def llm_model(system_msg, prompt_txt):
    """Send a prompt to IBM Granite LLM and return the response."""
    model = ModelInference(
        model_id="ibm/granite-4-h-small",
        project_id="skills-network",  # Replace with your own project_id
        credentials=Credentials(
            url="https://us-south.ml.cloud.ibm.com"
            # api_key="YOUR_API_KEY"  # Uncomment if running outside IBM Skills Network
        )
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": prompt_txt}
    ]
    return model.chat(messages=messages)["choices"][0]["message"]["content"]

# ── Pydantic Schema ────────────────────────────────────────────────────────────

class Restaurant(BaseModel):
    """Schema to validate structured restaurant JSON output."""
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)

# ── Prompt Templates ───────────────────────────────────────────────────────────

EXAMPLE_RESTAURANT_PARAGRAPH = restaurant_list[1]
EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
    }}
"""

def get_extraction_prompt(restaurant_paragraph):
    """Generate the prompt to extract restaurant data into JSON."""
    system_msg = f"""
    You are a data extraction assistant. Your job is to read a restaurant description
    and extract the information into a structured JSON format.

    Rules:
    - Always return valid JSON only, no extra text or explanation
    - For price_range, convert dollar signs to an integer ($ = 1, $$ = 2, $$$ = 3)
    - If a field has no information, use an empty list [] for arrays or null for strings
    - signatures and shortcomings should always be lists
    """
    user_prompt = f"""
    Task:
    Extract the restaurant information from the description below and return it as a JSON object
    with these exact fields: name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    """
    return system_msg, user_prompt


def get_repair_prompt(bad_output, error):
    """Generate the prompt to repair invalid JSON output."""
    system_msg = """
    You are a JSON repair expert. Your sole responsibility is to fix invalid or
    incorrectly formatted JSON outputs so they conform to the required schema.

    Rules:
    - Return only the corrected JSON, no explanations or extra text
    - Do not change any actual data values, only fix the formatting/structure
    - Ensure all required fields are present
    - price_range must be an integer ($ = 1, $$ = 2, $$$ = 3)
    - signatures and shortcomings must always be lists []
    """
    user_prompt = f"""
    The following JSON output is incorrect and failed schema validation.

    Wrong JSON output:
    {bad_output}

    Error message from validation:
    {error}

    Please fix the JSON output based on the error message above and return
    only the corrected valid JSON.
    """
    return system_msg, user_prompt

# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    structured_restaurant_lists = []

    for i, paragraph in enumerate(restaurant_list):

        # Step 1: Get initial output from LLM
        system_msg, user_prompt = get_extraction_prompt(paragraph)
        output = llm_model(system_msg, user_prompt)

        # Step 2: Validate and auto-repair if needed
        while True:
            try:
                parsed = json.loads(output)
                Restaurant.model_validate(parsed)
                break  # Valid — exit loop
            except (json.JSONDecodeError, ValidationError) as e:
                system_msg, user_prompt = get_repair_prompt(output, str(e))
                output = llm_model(system_msg, user_prompt)

        structured_restaurant_lists.append(output)

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(restaurant_list)} restaurants processed")

    # Step 3: Save to JSON file
    results = [json.loads(r) for r in structured_restaurant_lists]
    for i, r in enumerate(results):
        r['itemId'] = 1000001 + i

    output_path = "data/structured_restaurant_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\nDone! {len(results)} restaurants saved to {output_path}")
    print(f"\nSample output (first restaurant):")
    print(json.dumps(results[0], indent=4))


if __name__ == "__main__":
    main()