# restaurant_data_management.py
# Module 1 Lesson 3: Command-Line Restaurant Data Management UI

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

# ── Constants ──────────────────────────────────────────────────────────────────

FILEPATH = 'data/structured_restaurant_data.json'
BACKUP_PATH = 'data/structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating and **$** pricing, it is known for **beer-battered snapper tacos** and **zesty octopus ceviche**. The **salt-air energy** makes it a premier sun-drenched spot for open-air dining near the pier.'
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

# ── Pre-defined Helper Functions ───────────────────────────────────────────────

def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_data(data, file_path, backup_path):
    # Create a backup before writing
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def show_restaurant_card(res, index):
    """Displays restaurant data in a clean, vertical format."""
    print(f"\n{'='*15} RESTAURANT #{index} {'='*15}")
    name = res.get('name', res.get('restaurant_name', 'Unnamed Restaurant'))
    print(f"NAME : {name}")
    for key, value in res.items():
        if key.lower() not in ['name', 'restaurant_name']:
            label = key.replace('_', ' ').upper()
            print(f"{label:<12}: {value}")
    print('='*45)

# ── Pydantic Schema ────────────────────────────────────────────────────────────

class Restaurant(BaseModel):
    """The restaurant pydantic schema used in lesson 1."""
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

# ── Exercise 1: LLM Functions ──────────────────────────────────────────────────

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    You are a data extraction assistant. Your job is to read a restaurant description
    and extract the information into a structured JSON format.

    Rules:
    - Always return valid JSON only, no extra text or explanation
    - For price_range, convert dollar signs to an integer ($ = 1, $$ = 2, $$$ = 3)
    - If a field has no information, use an empty list [] for arrays or null for strings
    - signatures and shortcomings should always be lists
    """

    base_user_prompt = f"""
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
    return base_system_msg, base_user_prompt


def llm_model(system_msg, prompt_txt, params=None):
    model = ModelInference(
        model_id="ibm/granite-4-h-small",
        project_id="skills-network",
        credentials=Credentials(url="https://us-south.ml.cloud.ibm.com"),
        params=params
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]
    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]


def JSON_auto_repair_prompts(response, error_message):
    auto_repair_system_msg = """
    You are a JSON repair expert. Your sole responsibility is to fix invalid or
    incorrectly formatted JSON outputs so they conform to the required schema.

    Rules:
    - Return only the corrected JSON, no explanations or extra text
    - Do not change any actual data values, only fix the formatting/structure
    - Ensure all required fields are present
    - price_range must be an integer ($ = 1, $$ = 2, $$$ = 3)
    - signatures and shortcomings must always be lists []
    """
    auto_repair_prompt = f"""
    The following JSON output is incorrect and failed schema validation.

    Wrong JSON output:
    {response}

    Error message from validation:
    {error_message}

    Please fix the JSON output based on the error message above and return
    only the corrected valid JSON.
    """
    return auto_repair_system_msg, auto_repair_prompt


def new_data_entry_process(paragraph, itemId):
    """Use LLM pipeline to structure a new restaurant paragraph into JSON."""
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    output = llm_model(system_msg, user_prompt)

    # Validate and auto-repair loop
    while True:
        try:
            parsed = json.loads(output)
            Restaurant.model_validate(parsed)
            break  # Valid - exit loop
        except (json.JSONDecodeError, ValidationError) as e:
            repair_system_msg, repair_prompt = JSON_auto_repair_prompts(output, str(e))
            output = llm_model(repair_system_msg, repair_prompt)

    # Add itemId to the structured data
    result = json.loads(output)
    result['itemId'] = itemId
    return result

# ── Exercise 2: Main UI Function ───────────────────────────────────────────────

def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            # Iterate through records and show names
            for i, record in enumerate(data):
                name = record.get('name', 'N/A')
                print(f"[{i}] {name}")

        elif choice == '2':
            try:
                idx = int(input("Enter record index: "))
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1
                paragraph = input("Enter new restaurant description:\n")
                new_record = new_data_entry_process(paragraph, itemId)
                data.append(new_record)
                save_data(data, file_path, backup_path)
                print(" Restaurant added.")

            elif choice == '4':  # EDIT DATA
                try:
                    idx = int(input("Enter record index to edit: "))
                    if 0 <= idx < len(data):
                        for key in data[idx].keys():
                            new_value = input(f"{key} (current: {data[idx][key]}) — new value (Enter to skip): ")
                            if new_value.strip() != '':
                                data[idx][key] = new_value
                        save_data(data, file_path, backup_path)
                        print(" Record updated.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid index.")

            elif choice == '5':  # DELETE DATA
                try:
                    idx = int(input("Enter record index to delete: "))
                    if 0 <= idx < len(data):
                        data.pop(idx)
                        save_data(data, file_path, backup_path)
                        print(" Restaurant deleted.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid index.")

        elif choice == '6':  # EXIT
            break

        else:
            print("Invalid input.")

# ── Exercise 3: Unit Tests ─────────────────────────────────────────────────────

class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood, exposed brick, and Edison bulb lighting. Located in Asheville, NC, it has a 4.7/5 rating, $$$ pricing, and is known for its ramp gnocchi and pawpaw sorbet.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 2)
        self.assertIn(" Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', '1', '6']

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        mock_input.side_effect = ['5', 'no', '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()  # Unit Test
    # manage_restaurants(FILEPATH, BACKUP_PATH)  # Uncomment to run actual UI