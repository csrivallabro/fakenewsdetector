from fastapi import FastAPI, Body
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# Step 1: Load Hugging Face LLM (free, open source)
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype="auto",
    device_map="auto"
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=generator)

# Step 2: Define prompt template
template = """
You are a fact-checking assistant. Given a statement, determine if it is:
- TRUE if supported by facts
- FALSE if contradicted by facts
- UNVERIFIABLE if there is not enough information

Statement: "{statement}"

Answer with only one word: TRUE, FALSE, or UNVERIFIABLE.
"""

prompt = PromptTemplate(
    input_variables=["statement"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)

# Step 3: Create FastAPI app
app = FastAPI()

@app.post("/check_statement")
def check_statement(data: dict = Body(...)):
    statement = data.get("statement")
    if not statement:
        return {"error": "Please provide a statement."}
    
    result = chain.run(statement=statement)
    return {"statement": statement, "result": result.strip()}
