# module1_multimodal/process_multimodal_data.py
# Module 1 Lesson 2: Process multimodal data using LLaMA 4 vision model
# Generates image captions for food recipes and user reviews
# and enriches the existing JSON knowledge base with visual descriptions.

import json
import base64
import ast
import os
import requests
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Vision LLM ─────────────────────────────────────────────────────────────────

def vision_llm(system_msg, prompt_txt, image_path):
    """Send an image and text prompt to LLaMA 4 vision model and return caption."""
    model = ModelInference(
        model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
        project_id="skills-network",  # Replace with your own project_id
        credentials=Credentials(
            url="https://us-south.ml.cloud.ibm.com"
            # api_key="YOUR_API_KEY"  # Uncomment if running outside IBM Skills Network
        ),
        params={"max_tokens": 300}
    )

    with open(image_path, 'rb') as img_file:
        image_b64 = base64.b64encode(img_file.read()).decode('utf-8')

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                },
                {"type": "text", "text": prompt_txt}
            ]
        }
    ]

    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]

# ── Prompt Templates ───────────────────────────────────────────────────────────

def image_caption_prompt_template(food_name):
    """Generate prompt to caption a food recipe image."""
    system_msg = """You are a culinary expert and food writer.
    Your job is to describe food images in a concise, informative way focusing on
    ingredients, cooking style, presentation, and portion size."""

    prompt_txt = f"""Describe this image of {food_name}.
    Focus on visible ingredients, cooking method, presentation style, and portion size.
    Keep the description concise and under 3 sentences."""

    return system_msg, prompt_txt


def review_context_image_caption_prompt_template(reviews):
    """Generate prompt to caption a review image using review text as context."""
    system_msg = """You are a culinary expert and food critic.
    Your job is to describe food images in the context of user reviews.
    Generate concise, informative captions that align with the sentiment
    and details expressed in the reviews."""

    prompt_txt = f"""Describe this food image in the context of the following user review:

    Review: {reviews}

    Focus on visual elements that relate to the review such as presentation,
    atmosphere, and food quality. Keep it under 3 sentences."""

    return system_msg, prompt_txt

# ── URL Download with Retry ────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data_with_retry(url):
    """Download image from URL with automatic retries on failure."""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response

# ── Exercise 1: Recipe Image Captioning ───────────────────────────────────────

def process_recipe_data():
    """Load recipe data, generate image captions, and save enriched JSON."""
    print("Loading recipe data...")
    with open('data/Recipes.json', 'r') as f:
        recipe_data = json.load(f)
    print(f"Total recipes loaded: {len(recipe_data)}")

    print("\nGenerating captions for recipe images...")
    for i in range(len(recipe_data)):
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(recipe_data)} recipes processed")

        image_path = f"data/synthetic_recipe_images/recipe{recipe_data[i]['id']}.png"
        system_msg, prompt_txt = image_caption_prompt_template(recipe_data[i]['name'])
        response = vision_llm(system_msg, prompt_txt, image_path)
        recipe_data[i]['image_description'] = response

    output_path = 'data/augmented_food_recipe.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recipe_data, f, indent=4)

    print(f"\nDone! Saved to {output_path}")
    print("\nSample caption (first recipe):")
    print(f"{recipe_data[0]['name']}: {recipe_data[0]['image_description']}")

# ── Exercise 2: User Review Image Captioning ──────────────────────────────────

def process_review_data():
    """Load user review data, generate image captions, and save enriched JSON."""
    print("Loading user review data...")
    with open('data/Synthetic-User-Reviews.json', 'r') as f:
        user_review_data = json.load(f)
    print(f"Total reviews loaded: {len(user_review_data)}")

    print("\nGenerating captions for review images...")
    for i in range(len(user_review_data)):
        review_images = ast.literal_eval(user_review_data[i]['images'])

        review_image_captions = []
        if len(review_images) > 0:
            for img_url in review_images:
                try:
                    image_data = get_data_with_retry(img_url)
                    print(f"Downloaded image for review {i + 1}")
                except Exception as e:
                    print(f"Failed at url {img_url}: {e}")
                    continue

                with open('data/review_image_placeholder.jpg', 'wb') as img_file:
                    img_file.write(image_data.content)

                reviews = user_review_data[i]['text']
                system_msg, prompt_txt = review_context_image_caption_prompt_template(reviews)
                response = vision_llm(system_msg, prompt_txt, 'data/review_image_placeholder.jpg')
                review_image_captions.append(response)

        user_review_data[i]['image_captions'] = review_image_captions

    output_path = 'data/augmented_user_review.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(user_review_data, f, indent=4)

    print(f"\nDone! Saved to {output_path}")
    print("\nSample caption (first review):")
    print(f"Review: {user_review_data[0]['title']}")
    print(f"Caption: {user_review_data[0]['image_captions'][0]}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 50)
    print("Module 1 Lesson 2: Process Multimodal Data")
    print("=" * 50)

    print("\n--- Exercise 1: Recipe Image Captioning ---")
    process_recipe_data()

    print("\n--- Exercise 2: User Review Image Captioning ---")
    process_review_data()

    print("\nAll done! Both JSON files saved to data/ folder.")


if __name__ == "__main__":
    main()
