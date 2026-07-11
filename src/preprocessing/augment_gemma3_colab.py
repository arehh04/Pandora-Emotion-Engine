import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from tqdm import tqdm

def main():
    print("Initializing Qwen 2.5-1.5B-Instruct for Stratified Data Augmentation...")
    
    # 1. Load Data
    # Assuming data is in the same directory or adjust path for Colab
    data_path = "train_clean.csv" 
    tokens_path = "train_tokens.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Make sure you upload train_clean.csv and train_tokens.csv to Colab.")
        return
        
    df_clean = pd.read_csv(data_path)
    df_tokens = pd.read_csv(tokens_path)
    
    # Merge to get texts and extraversion scores in one place
    df = df_clean.copy()
    df['extraversion'] = df_tokens['extraversion']
    
    # 1.5 Load Checkpoint if resuming
    if os.path.exists("train_augmented_checkpoint.csv"):
        print("Found existing checkpoint! Resuming progress...")
        checkpoint_df = pd.read_csv("train_augmented_checkpoint.csv")
        # Ensure we don't duplicate rows if we run it multiple times
        # The checkpoint contains ONLY the augmented rows
        df = pd.concat([df, checkpoint_df], ignore_index=True)
    
    # 2. Stratified Binning
    print("\n--- Stratified Binning ---")
    bins = [0, 33, 66, 100]
    labels = ['Low', 'Medium', 'High']
    df['expressiveness_bin'] = pd.cut(df['extraversion'], bins=bins, labels=labels, include_lowest=True)
    
    counts = df['expressiveness_bin'].value_counts()
    print("Current Distribution (Including any loaded checkpoints):")
    print(counts)
    
    majority_count = counts.max()
    minority_bins = counts[counts < majority_count].index.tolist()
    
    # 3. Load Model (Gemma 3 - Quantized for Colab)
    model_name = "google/gemma-3-4b-it" # or gemma-3-12b-it if memory allows
    print(f"\nLoading model {model_name} in 4-bit quantization...")
    
    # You must install these in Colab first:
    # !pip install -U transformers bitsandbytes accelerate
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" # Crucial for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 4. Augmentation Logic
    augmented_rows = []
    
    system_prompt = """You are a professional linguistic rewriter.

Rewrite the sentence while preserving:
- original meaning
- emotional tone (important for social behavior signal)
- personality signal (introvert / extrovert expression style)
- intensity level

Rules:
- Do NOT change meaning
- Do NOT add new information
- Keep same sentiment direction
- Keep same social behavior style
- Only change sentence structure and wording

Return EXACTLY 3 paraphrased versions separated by newlines. Do not include numbering, explanations, or quotes. Make sure paraphrase preserves behavior traits. Example:
Input: "I prefer staying at home and avoiding crowded places"
Output:
I usually like staying indoors and avoid busy environments
I tend to choose staying in rather than navigating crowds
Being at home away from densely packed areas is my preference"""

    def generate_paraphrases_batch(texts):
        text_inputs = []
        for text in texts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Input:\n"{text}"'}
            ]
            text_inputs.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
            
        model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        batch_results = []
        for response in responses:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            batch_results.append(lines[:3])
        return batch_results

    print("\n--- Starting LLM Augmentation ---")
    BATCH_SIZE = 8
    
    # Simple memory cache for identical sentences
    paraphrase_cache = {}
    
    for b in minority_bins:
        current_count = counts[b]
        needed = majority_count - current_count
        print(f"Bin '{b}' needs {needed} more samples.")
        
        bin_df = df[df['expressiveness_bin'] == b]
        samples_generated = 0
        pbar = tqdm(total=needed, desc=f"Augmenting {b}")
        
        batch_rows = []
        while samples_generated < needed:
            for _, row in bin_df.sample(frac=1).iterrows():
                if samples_generated >= needed:
                    break
                    
                batch_rows.append(row)
                
                if len(batch_rows) == BATCH_SIZE:
                    texts = [r['bert_text'] for r in batch_rows]
                    
                    # Separate cached vs uncached
                    texts_to_generate = [t for t in texts if t not in paraphrase_cache]
                    
                    try:
                        # Only run model on texts we haven't seen before
                        if texts_to_generate:
                            generated_results = generate_paraphrases_batch(texts_to_generate)
                            for t, res in zip(texts_to_generate, generated_results):
                                paraphrase_cache[t] = res
                                
                        # Now assemble all results using the cache
                        batch_paraphrases = [paraphrase_cache[t] for t in texts]
                        
                        for r, paraphrases in zip(batch_rows, batch_paraphrases):
                            for p in paraphrases:
                                if samples_generated >= needed:
                                    break
                                
                                new_row = r.copy()
                                new_row['bert_text'] = p
                                
                                if 'type' in new_row:
                                    new_row['type'] = str(r['type']) + '_AUGMENTED'
                                else:
                                    new_row['is_augmented'] = True
                                    
                                augmented_rows.append(new_row)
                                samples_generated += 1
                                pbar.update(1)
                                
                                if samples_generated % 100 == 0:
                                    # Save just the newly generated rows to a temp file, we will merge them later if needed
                                    pd.DataFrame(augmented_rows).to_csv("train_augmented_temp.csv", index=False)
                    except Exception as e:
                        print(f"Batch generation error - {e}")
                    
                    batch_rows = []
                    
        pbar.close()

    # Append the newly generated rows to the main dataframe
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        final_df = pd.concat([df, aug_df], ignore_index=True)
    else:
        final_df = df

    print("\n--- Final Balanced Distribution ---")
    print(final_df['expressiveness_bin'].value_counts())
    
    # Clean up the bin column before saving
    final_df.drop(columns=['expressiveness_bin'], inplace=True)
    
    # Save the final file
    print("\nSaving train_augmented.csv ...")
    final_df.to_csv("train_augmented.csv", index=False)
    
    # Also update the checkpoint so it contains everything (for future resumes)
    # We only want to save the augmented rows in the checkpoint, not the original data
    checkpoint_out = final_df[final_df.get('is_augmented', pd.Series(False, index=final_df.index)) | final_df['type'].astype(str).str.contains('AUGMENTED', na=False)]
    checkpoint_out.to_csv("train_augmented_checkpoint.csv", index=False)
    
    print("Done! You can now use train_augmented.csv for feature extraction.")

if __name__ == "__main__":
    main()
