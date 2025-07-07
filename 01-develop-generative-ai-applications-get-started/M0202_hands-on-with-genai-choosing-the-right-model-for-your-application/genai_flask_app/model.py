from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import PromptTemplate
from config import PARAMETERS, CREDENTIALS, AIRX_MODEL_ID, PLUS4_MODEL_ID, FLASH_MODEL_ID

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

#Define JSON output structure
# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str = Field(description="Category of the inquiry (e.g., billing, technical, general)")
    action: str = Field(description="Recommended action for the support rep")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)


def initialize_model(model_id):
    model = ChatZhipuAI(
        model_id=model_id,
        parameters=PARAMETERS,
        credentials=CREDENTIALS,
    )
    return model

# Initialize models
airx_model = initialize_model(AIRX_MODEL_ID)
plus4_model = initialize_model(PLUS4_MODEL_ID)
flash_model = initialize_model(FLASH_MODEL_ID)

# Prompt template
airx_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "user_prompt"]
)

plus4_template = PromptTemplate(
    template="<|system|>{system_prompt}\n\<|user|>{user_prompt}\n<|assistant|>",
    input_variables=["system_prompt", "user_prompt"]
)

flash_template = PromptTemplate(
    template="<s>[INST]{system_prompt}\n{user_prompt}[/INST]",
    input_variables=["system_prompt", "user_prompt"]
)

def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({'system_prompt':system_prompt, 'user_prompt':user_prompt})

# Model-specific response functions
def airx_response(system_prompt, user_prompt):
    return get_ai_response(airx_model, airx_template, system_prompt, user_prompt)

def plus4_response(system_prompt, user_prompt):
    return get_ai_response(plus4_model, plus4_template, system_prompt, user_prompt)

def flash_response(system_prompt, user_prompt):
    return get_ai_response(flash_model, flash_template, system_prompt, user_prompt)

