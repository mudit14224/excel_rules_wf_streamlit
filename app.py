import streamlit as st
import pandas as pd
import json
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, create_model
from typing import Optional
from datetime import date
from langchain.schema import HumanMessage, SystemMessage
from utils import extract_code_block
from dotenv import load_dotenv
from prompts import pydantic_prompt_template, extract_template

load_dotenv()

model = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-12-01-preview",  
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# --- Streamlit UI ---
st.title("Excel → Structured JSON via LLMs 🧠")

uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
instruction = st.text_area("Describe the output you want", height=150)

run = st.button("Generate JSON")

if uploaded_file and instruction and run:
    df = pd.read_excel(uploaded_file)
    st.success("Excel loaded successfully.")

    # Prompt LLM to generate the Pydantic model class
    with st.spinner("Generating schema..."):
        pydantic_prompt = PromptTemplate(input_variables=['columns', 'instruction'], template=pydantic_prompt_template)
        # Format the prompt with actual values
        formatted_prompt = pydantic_prompt.format(
            columns=df.columns.tolist(),
            instruction=instruction
        )

        messages = [
            SystemMessage(content="You are a senior Python engineer skilled in writing typed Pydantic models."),
            HumanMessage(content=formatted_prompt)
        ]

        response = model.invoke(messages)

        code_block = extract_code_block(response.content)

        # Full set of globals needed by the code
        safe_globals = {
            "BaseModel": BaseModel,
            "Optional": Optional,
            "date": date  
        }

        # Exec with all required symbols
        namespace = {}
        exec(code_block, safe_globals, namespace)

        # Extract the model
        OrderModel = namespace["Order"]

    structured_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-12-01-preview",
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],  
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2).with_structured_output(OrderModel)

    row_prompt = PromptTemplate(
        input_variables=["row_description"],
        template=extract_template)
    
    structured_outputs = []
    with st.spinner("Processing rows..."):
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            prompt_text = row_prompt.format(row_description=row_dict)
            try:
                structured = structured_llm.invoke(prompt_text)
                structured_outputs.append(structured)
            except Exception as e:
                st.warning(f"Error on row: {e}")

    st.success("Done!")

    # Show result
    st.subheader("Structured Output")
    

    # Convert each Pydantic object to dict
    json_ready = [obj.model_dump() for obj in structured_outputs]

    # Dump to JSON string
    json_output = json.dumps(json_ready, indent=2)

    st.json(json_output)