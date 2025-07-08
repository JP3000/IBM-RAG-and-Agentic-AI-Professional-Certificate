# Import the necessary packages
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_community.chat_models import ChatZhipuAI
import os

# Specify the model and project settings 
# (make sure the model you wish to use is commented out, and other models are commented)
#model_id = 'mistralai/mixtral-8x7b-instruct-v01' # Specify the Mixtral 8x7B model
# model_id = 'ibm/granite-3-3-8b-instruct' # Specify IBM's Granite 3.3 8B model

model_id = 'glm-4-plus'

# Set the necessary parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # Specify the max tokens you want to generate
    GenParams.TEMPERATURE: 0.5, # This randomness or creativity of the model's responses
}

# project_id = "skills-network"

# Wrap up the model into WatsonxLLM inference
Zhipu_llm = ChatZhipuAI(
    model_id=model_id,
    api_key=os.getenv("ZHIPUAI_API_KEY"),  # Ensure you have set your ZhipuAI API key in the environment variables
    params=parameters,
)

# Get the query from the user input
query = input("Please enter your query: ")

# Print the generated response
print(Zhipu_llm.invoke(query))