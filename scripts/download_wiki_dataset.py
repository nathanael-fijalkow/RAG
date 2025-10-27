import os
import sys
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "HuggingFaceFW/finewiki"
SPLIT = "train"
TARGET_LANGUAGE = "en"
NUM_ENTRIES = 100
OUTPUT_DIR = "data/docs/wiki_finewiki_en"

def load_and_save_entries():
    """
    Loads entries from the FineWiki dataset using streaming to avoid loading the
    entire dataset into memory, filters for English, and saves the first N.
    """
    print(f"Loading dataset in streaming mode: {DATASET_NAME} (Split: {SPLIT}).")
    print(f"Will retrieve up to {NUM_ENTRIES} entries.")

    # 1. Load the dataset in streaming mode (critical for avoiding full load)
    streaming_dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT,
        streaming=True
    )

    # 2. Setup tracking variables
    processed_count = 0
    checked_count = 0
    max_to_check = NUM_ENTRIES * 10_000  # Stop after checking 10,000 entries times target to avoid infinite loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Iterate over the streaming dataset, filter, and save until the limit is reached
    for entry in streaming_dataset:
        checked_count += 1
        
        # Safety limit: stop checking after max_to_check entries
        if checked_count > max_to_check:
            print(f"\n\nReached maximum check limit ({max_to_check} entries). Stopping.")
            if processed_count == 0:
                print("\nWarning: No entries were found in the stream.")
            return

        if entry["in_language"] == TARGET_LANGUAGE and "film" in entry["url"] and "in" not in entry["url"]:            
            file_id = entry['id'].replace('/', '_')
            filename = os.path.join(OUTPUT_DIR, f"{file_id}.md")

            content = entry['text']

            markdown_content = (
                f"# FineWiki Entry: {entry['title']}\n\n"
                f"**ID:** `{entry['id']}`\n"
                f"**URL:** <{entry['url']}>\n\n"
                f"--- \n\n"
                f"{content}"
            )

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            processed_count += 1

            # Check if we have reached the desired limit and break the stream
            if processed_count >= NUM_ENTRIES:
                print("\nProcess complete!")
                print(f"Successfully saved {processed_count} files in the '{OUTPUT_DIR}' directory.")
                return

if __name__ == "__main__":
    load_and_save_entries()
