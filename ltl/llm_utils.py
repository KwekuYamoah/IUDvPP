import re
import openai


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
    gpt_model = "gpt-4o" # gpt-4 or gpt-4o
    print(f"\t\tModel: {gpt_model} || Deterministic?: {deterministic}")
    if deterministic == True:
        completion = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[{
                "role": "user",
                "content":  user_prompt
            }],
            temperature=0,  # use deterministic greedy token decoding during evaluation experiments however deterministic decoding is not guaranteed, hence peform multiple runs and get mode of the responses
            max_tokens=2000,
        )
    else:
        completion = openai.ChatCompletion.create(
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


def limp_llm_plan(limp_plan_incontext_prompt, instruction, intent_dict, verbose=False):
    prompt_meat = f'''
    Input_instruction:  {instruction}
    Goal_intents: {intent_dict['Goal_intent']},
    Avoidance_intent: {intent_dict['Avoidance_intent']},
    Detail_intent: {intent_dict['Detail_intent']}
    Output: '''
    complete_prompt = limp_plan_incontext_prompt + prompt_meat
    print(f"Prompt: {complete_prompt}") if verbose else None
    response = generate_response_from_gpt4(complete_prompt)
    return response

