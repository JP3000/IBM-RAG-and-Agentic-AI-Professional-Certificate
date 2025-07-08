# Import the necessary packages
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_community.chat_models import ChatZhipuAI
import os
import gradio as gr

model_id = 'glm-4-plus'

# Set the necessary parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # Specify the max tokens you want to generate
    GenParams.TEMPERATURE: 0.5, # This randomness or creativity of the model's responses
}

Zhipu_llm = ChatZhipuAI(
    model_id=model_id,
    api_key=os.getenv("ZHIPUAI_API_KEY"),  # Ensure you have set your ZhipuAI API key in the environment variables
    params=parameters,
)

# Function to generate a response from the model
def generate_response(prompt_txt):
    generated_response = Zhipu_llm.invoke(prompt_txt)
    return generated_response
# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Zhipu.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)
# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860)

