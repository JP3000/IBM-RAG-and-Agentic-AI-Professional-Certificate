from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextModerationsMetaNames
from langchain_community.chat_models import ChatZhipuAI
import os
from langchain_core.messages import HumanMessage
model_id = 'glm-4-plus'

credentials = Credentials(
    api_key = os.getenv("ZHIPUAI_API_KEY"),
)

model = ChatZhipuAI(
    credentials=credentials,
    model_id=model_id,
    max_tokens=512,
)

text = """ 
Only reply with the answer. What is the capital of Canada?
"""

result = model.generate([[HumanMessage(content=text)]])
print(result.generations[0][0].text)