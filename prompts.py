pydantic_prompt_template = """
You are an expert Python developer.

Given the following DataFrame columns:
{columns}

And this user instruction:
{instruction}

Generate a Python class using Pydantic's BaseModel named `Order`.

Requirements:
- Only include fields explicitly required by the user instruction.
- Do not infer or add any extra fields, even if they are mentioned in the dataset columns.
- Do not include any validators, computed properties, or transformation logic.
- Instead, include a docstring in the class with a clear description of the transformation logic or mapping (e.g., "Convert all amounts to USD", "status is true or false").
- Use accurate field types (e.g., int, str, float). Make fields Optional only if the instruction implies it.
- Return only the code — no explanation or markdown.
"""


extract_template = """
You are extracting structured information from tabular data.

Here is a row from the dataset:
{row_description}

Extract and return a JSON object matching the required schema.
""" 