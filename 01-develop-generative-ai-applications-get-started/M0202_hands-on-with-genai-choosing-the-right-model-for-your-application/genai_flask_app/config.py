from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import os

# Model parameters
PARAMETERS = {
    # GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 256,
}

# watsonx credentials
# Note: Normally we'd need an API key, but in Skill's Network Cloud IDE will automatically handle that for you.
CREDENTIALS = {
    # "url": "https://us-south.ml.cloud.ibm.com",
    # "project_id": "skills-network"
    "api_key": os.getenv("ZHIPUAI_API_KEY"),
}

# Model IDs
# LLAMA3_MODEL_ID = "meta-llama/llama-3-2-11b-vision-instruct"
# GRANITE_MODEL_ID = "ibm/granite-3-8b-instruct"
# MIXTRAL_MODEL_ID = "mistralai/mistral-large"

AIRX_MODEL_ID = "glm-z1-airx",
PLUS4_MODEL_ID = "glm-4-plus",
FLASH_MODEL_ID = "glm-z1-flash",