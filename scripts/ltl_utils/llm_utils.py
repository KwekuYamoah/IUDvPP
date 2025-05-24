import re
import json
from jsonschema import validate, ValidationError
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
    
    gpt_model = "o1" # gpt-4 or gpt-4o or o3-mini
    print(f"\t\tModel: {gpt_model} || Deterministic?: {deterministic}")
    if deterministic == True:
        completion = client.chat.completions.create(
            model=gpt_model,
            reasoning_effort= 'medium', # for o series of models, comment out when not using this
            messages=[{
                "role": "user",
                "content":  user_prompt
            }],
            # temperature=0,  # use deterministic greedy token decoding during evaluation experiments however deterministic decoding is not guaranteed, hence peform multiple runs and get mode of the responses
            #max_tokens=2000,
            max_completion_tokens=2000,
            
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
                        "description": "A list of resolved referents with their attributes."
                    },
                    "TaskPlan": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A sequence of tasks to achieve the goal."
                    }
                },
                "required": ["Interpretation", "Explanation", "ResolvedReferents", "TaskPlan"],
                "additionalProperties": False
            }
    
    # More specific system message to prevent hallucination
    system_message = (
        "You are an intelligent assistant specialized in interpreting tasks and generating structured task plans. "
        "Your responses must strictly adhere to the provided JSON schema without adding any extra information, explanations, or deviations. "
        "Do not include anything outside of the JSON schema, and make sure all fields are present. "
        "The model output has all the information needed to create the JSON schema, so outside of this, nothing extra should be added. "
        "If you do not have enough information to complete a field, return an empty string for that field instead of guessing or hallucinating. "
        "Be concise and precise, and do not add any extraneous information."
    )

    if deterministic:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": model_output
                }
            ],
            temperature=0,  # Lower temperature for deterministic output
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
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": model_output
                }
            ],
            temperature=0.3,  # Slightly higher temperature for creativity, but still controlled
            top_p=0.9,  # Adjust top_p to reduce the chance of hallucinations
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



def limp_llm_plan(limp_plan_incontext_prompt, instruction, task_plans, intent_dict, baseline=None, verbose=False):
    print(f"Executing Baseline: {baseline}")

    if "intent_label" in baseline:
        prompt_meat = f'''
        Input_instruction:  {instruction}
        Goal_intents: {intent_dict['Goal_intent']}
        Avoidance_intent: {intent_dict['Avoidance_intent']}
        Detail_intent: {intent_dict['Detail_intent']}
        Possible_plans: {task_plans}
        Output: '''
    elif "asr" in baseline:
        prompt_meat = f'''
        Input_instruction:  {instruction}
        Possible_plans: {task_plans}
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

