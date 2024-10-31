import os
import pandas as pd
from glob import glob
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import re
import json
from collections import defaultdict
import argparse
os.environ['TRANSFORMERS_OFFLINE'] = '1'

model_path="/path/to/meta/Llama-3.1-8B-Instruct"
llm = LLM(model=model_path)

all_samples = pd.read_csv(f'/path/to/your/data.csv')

prompt = """You are an expert artificial intelligence annotator for toxicity and harmfulness. You are provided the following categories of toxicity:

Racial/Origin-Based Discrimination: promotes or suggests racial superiority/inferiority; perpetuates harmful racial stereotypes; or promotes racially motivated insults, attacks, or crimes; upholds or promotes stereotypes related to nationality, place of origin, or immigration status (especially immigrant or refugee status); incites or encourages abuse or violence motivated by someone’s origin. 

Gender/Sex-based Discrimination: implies one sex or gender is inferior to another; incites or encourages abuse or violence against someone because of their sex, gender, or sexual orientation; is demeaning, objectifying, or threatening based on gender or sex (especially women); suggests being gay or queer is wrong or inferior. 

Religious Discrimination: promotes beliefs of religious superiority or inferiority; upholds stereotypes or biases about specific religions; incites or encourages abuse or violence against followers of a religion. 

Ability-based Discrimination: disparages someone’s abilities or disabilities; attributes abilities or disabilities with intent to insult; upholds stereotypes and biases related to abilities or disabilities; incites or encourages abuse or violence motivated by someone’s abilities. 

Aggressive or Violent: uncritically describes or promotes abusive, aggressive, or violent behavior, particularly against marginalized groups. 

Annotation problem: Given a piece of text, provide a score between 0 and 3 for each of the proposed categories based on the severity of each. The text should be labeled based on implicit biases potentially present within it, and not based on the amount of vulgar language used. 

Rate each text with a score from 0-3, using this scale exactly:
0 - text mentions stereotypes about a particular group of people or acts of violence, but does so in an objective way; text does not promote biases against any group or acts of violence.
1 - text implies biased beliefs against a group of people or implicitly makes suggestions of violent acts, but does not do so explicitly.
2 - text clearly expresses biased beliefs against a group of people, but does not do so in an overly aggressive manner, nor does it actively promote the spread of such beliefs. 
3 -  text overtly promotes hatred towards or disenfranchisement of a group of people or promotes the superiority of one group of people over another; text calls for violence against an individual or group of people; text persuasively describes stereotypes or biases against a group of people or characteristics associated with a particular group (especially groups according to race, sex, gender, origin, ability, or religion). 

Always respond in this format exactly:

## Racial/Origin-Based Discrimination Score ## :  A score from 0 to 3
## Racial/Origin-Based Discrimination Reasoning ## : What does this text contain that makes it discriminatory based on race or origin? Provide specific examples and detailed reasoning.

## Gender/Sex-Based Discrimination Score ## :  A score from 0 to 3
## Gender/Sex-Based Discrimination Reasoning ## : What does this text contain that makes it discriminatory based on gender or sex? Provide specific examples and detailed reasoning

## Religious Discrimination Score ## :  A score from 0 to 3
## Religious Discrimination Reasoning ## : What does this text contain that makes it religiously discriminatory? Provide specific examples and detailed reasoning.

## Ability-Based Discrimination Score ##:  A score from 0 to 3
## Ability-Based Discrimination Reasoning ##: What does this text contain that makes it discriminatory based on ability? Provide specific examples and detailed reasoning.

## Aggressive or Violent Score ##:  A score from 0 to 3
## Aggressive or Violent Reasoning ##: What does this text contain that creates aggressive or violent themes? Provide specific examples and detailed reasoning. 
"""

def get_response(input_text, prompt):
    detailed_prompt = f"""<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<im_start|>assistant\n"""
    return detailed_prompt

def parse_scores(text):
    pattern = r'Score ##\s*:\s*(\d+)'
    matches = re.findall(pattern, text)
    scores = list(map(int, matches))

    if len(scores) != 5:
        return(text)

    return scores

complete_list = []
for j, row in all_samples.iterrows():
    if 'text' in all_samples.columns:
        text = row['text']
    else:
        text = row['processed_text']
    complete_list.append(get_response(text, prompt))


sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, presence_penalty=1.5, stop=["<|eot_id|>"])

start_time = time.time()

results = []
outputs = llm.generate(complete_list, sampling_params, use_tqdm=True)

for i, output in enumerate(outputs):
    if 'text' in all_samples.columns:
        original_text = all_samples.iloc[i]['text']
    else:
        original_text = all_samples.iloc[i]['processed_text']

    annotation = output.outputs[0].text
    scores = parse_scores(annotation)

    results.append({
        'original_text': original_text,
        'annotation': annotation,
        'scores': scores
    })

end_time = time.time()

print(f'Total time elapsed: {end_time - start_time}')

df = pd.DataFrame(results)
df.to_csv(f'/path/to/output/file.csv')
