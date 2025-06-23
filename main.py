#!/usr/bin/env -S uv run

import os
import time
import pandas as pd
from openai import OpenAI

from subskills import subskills
from secrets import OPENAI_API_KEY


# Load the CSV data globally
df = pd.read_csv("data.csv")

def generate_prompt(essay, target_text, subskill) -> tuple[str, str]:
    """Generate system and user prompts for discourse classification"""
    system_prompt = open("prompts_cur/system", "r").read()
    user_prompt_template = open("prompts_cur/user", "r").read()
    
    user_prompt = user_prompt_template.replace("{essay}", essay).replace("{target_text}", target_text).replace("{subskill}", subskill)
    
    return system_prompt, user_prompt

def get_essay_for_id(essay_id) -> str:
    """Retrieve the full essay text for a given essay ID"""
    essay_rows = df[df['essay_id'] == essay_id]
    
    if essay_rows.empty:
        return f"Essay {essay_id} not found"
    
    essay_texts = essay_rows.sort_values('discourse_id')['discourse_text'].tolist()
    full_essay = " ".join(essay_texts)
    
    return full_essay

def classify_discourse_element(essay_id, subskill, discourse_text) -> str:
    """Classify a discourse element using OpenAI API"""

    essay = get_essay_for_id(essay_id)
    sys_prompt, user_prompt = generate_prompt(essay, discourse_text, subskill)
    
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.responses.create(
        model="gpt-4.1",
        instructions=sys_prompt,
        input=user_prompt
    )
    
    return response.output_text

def main():
    """Main function to process the CSV data"""
    print("Starting discourse element classification...")
    print(f"Processing {len(df)} rows from data.csv")

    
    max_cnt = 200
    for idx, row in df.iterrows():
        if idx >= max_cnt:
            break
            
        print(f"\nProcessing row {idx + 1}/{max_cnt}")
        print(f"Discourse ID: {row['discourse_id']}")
        print(f"Essay ID: {row['essay_id']}")
        print(f"Discourse Text: {row['discourse_text'][:100]}...")  # Show first 100 chars
        

        for subskill_name, subskill in subskills.items():
            if row["discourse_type"] not in ["Claim", "Lead"]:
                classification = "Null"
            else:
                classification = classify_discourse_element(
                    essay_id=row['essay_id'],
                    subskill=subskill,
                    discourse_text=row['discourse_text']
                )
        
            df.at[idx, float(subskill_name)] = classification
            print(f"Classification ({subskill_name}): {classification}")
        
        time.sleep(0.5)
    
    output_file = "data_with_classifications.csv"
    df_output = df.head(100).copy()  # Only save the processed rows
    df_output.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
