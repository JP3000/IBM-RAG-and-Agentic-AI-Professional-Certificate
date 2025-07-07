from model import airx_response, plus4_response, flash_response

def call_all_models(system_prompt, user_prompt):
    airx_result = airx_response(system_prompt, user_prompt)
    plus4_result = plus4_response(system_prompt, user_prompt)
    flash_result = flash_response(system_prompt, user_prompt)

    print("AirX Response:\n", airx_result.content)
    print("\nPlus4 Response:\n", plus4_result.content)
    print("\nFlash Response:\n", flash_result.content)

# Example call to test all models
call_all_models("You are a helpful assistant who provides concise and accurate answers", "What is the capital of Canada? Tell me a cool fact about it as well")

