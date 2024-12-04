import re
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# set openai client with api key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_intent_dict(intent_labels):
    intent ={
        "Goal_intent":[],
        "Avoidance_intent": [],
        "Detail_intent": []
    }
    for word, label in intent_labels.items():
        if label == 1:
            intent["Goal_intent"].append(word)
        elif label == 2:
            intent["Detail_intent"].append(word)
        elif label == 3:
            intent["Avoidance_intent"] = word
    return intent

def generate_response_from_gpt4(user_prompt=None, deterministic=True):
    
    gpt_model = "gpt-4" # gpt-4 or gpt-4o
    print(f"\t\tModel: {gpt_model} || Deterministic?: {deterministic}")
    if deterministic == True:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[{
                "role": "user",
                "content":  user_prompt
            }],
            temperature=0,  # use deterministic greedy token decoding during evaluation experiments however deterministic decoding is not guaranteed, hence peform multiple runs and get mode of the responses
            max_tokens=2000,
            
        )
    else:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[{
                "role": "user",
                "content":  user_prompt
            }],
            temperature=0.3,
            top_p=1,
            max_tokens=2000,
        )

    response = completion.choices[0].message.content
    return response

def generate_json_with_gpt(model_output, deterministic=True):
    # use 4o variant of gpt
    gpt_model = "gpt-4o"
    print(f"\t\tModel: {gpt_model} || Deterministic?: {deterministic}")

    # define json schema
    task_schema = {
                "type": "object",
                "properties": {
                    "Interpretation": {
                        "type": "string",
                        "description": "A natural language interpretation of the task."
                    },
                    "Explanation": {
                        "type": "string",
                        "description": "A detailed explanation of how the interpretation was derived."
                    },
                    "ResolvedReferents": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A list of resolved referents with their attributes. Example ['mug::ison(table::isnear(sink))']"
                    },
                    "TaskPlan": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A sequence of tasks to achieve the goal. Example:[ ‘goto[mug::ison(table::isnear(sink))]’, ‘pick[mug::ison(table::isnear(sink))]’, ‘move[mug::ison(table::isnear(sink))] to new location’ ]"
                    }
                },
                "required": ["Interpretation", "Explanation", "ResolvedReferents", "TaskPlan"],
                "additionalProperties": False
            }
    if deterministic == True:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent assistant specialized in interpreting tasks and generating structured task plans. Your responses must strictly adhere to the provided JSON schema without adding any extra information, explanations, or deviations. Ensure that all required fields are present and correctly formatted as per the schema."
                },
                {
                    "role": "user",
                    "content": model_output
                }
            ],
            temperature=0,  # use deterministic greedy token decoding during evaluation experiments however deterministic decoding is not guaranteed, hence peform multiple runs and get mode of the responses
            max_tokens=2000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "task_planner_schema",
                    "schema": task_schema,
                    "strict": True
                }
            }
            
        )
    else:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent assistant specialized in interpreting tasks and generating structured task plans. Your responses must strictly adhere to the provided JSON schema without adding any extra information, explanations, or deviations. Ensure that all required fields are present and correctly formatted as per the schema."
                },
                {
                    "role": "user",
                    "content": "Pick up the book (which is on the table) and that book has a red cover."
                }
            ],
            temperature=0.3,
            top_p=1,
            max_tokens=2000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "task_planner_schema",
                    "schema": task_schema,
                    "strict": True
                }
            }
        )
    
    response = completion.choices[0].message.content
    return response

def limp_llm_plan(limp_plan_incontext_prompt, instruction, intent_dict, baseline=None, verbose=False):
    print(f"Executing Baseline: {baseline}")

    if "intent_label" in baseline:
        prompt_meat = f'''
        Input_instruction:  {instruction}
        Goal_intents: {intent_dict['Goal_intent']}
        Avoidance_intent: {intent_dict['Avoidance_intent']}
        Detail_intent: {intent_dict['Detail_intent']}
        Output: '''
    elif "asr" in baseline:
        prompt_meat = f'''
        Input_instruction:  {instruction}
        Output: '''
    elif "prosody_value" in baseline:
        prompt_meat = f'''
        Input_instruction:  {instruction}
        prosody_values: 
        Output: '''
    else:
        #error exit
        print("Error: Baseline not found")
        return None

    complete_prompt = limp_plan_incontext_prompt + prompt_meat
    print(f"Complete prompt:\n {complete_prompt}") if verbose else None
    response = generate_response_from_gpt4(complete_prompt)
    return response

