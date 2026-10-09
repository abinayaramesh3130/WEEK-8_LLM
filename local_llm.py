
import pandas as pd
import ollama
import os

# Load dataset
input_file = r"C:\Users\admin\OneDrive\Desktop\LLM local_inference\dataset\prompt.csv.xlsx"

prompts = pd.read_excel(input_file)
prompts.columns = prompts.columns.str.strip().str.lower()

if "prompt" not in prompts.columns:
    raise ValueError("Excel file must contain a 'prompt' column.")

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

model = "llama3.2"

responses = []
comparison_data = []
conversation_lines = []

# Generate responses
for index, row in prompts.iterrows():

    prompt_id = row["prompt_id"] if "prompt_id" in prompts.columns else index + 1
    domain = row["domain"] if "domain" in prompts.columns else "General"
    prompt = str(row["prompt"])

    print(f"\nUser: {prompt}")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    print(f"Assistant: {answer}")

    # Calculate response length
    word_count = len(answer.split())
    character_count = len(answer)

    responses.append({
        "prompt_id": prompt_id,
        "domain": domain,
        "prompt": prompt,
        "response": answer
    })

    # Comparison data
    comparison_data.append({
        "prompt_id": prompt_id,
        "domain": domain,
        "prompt": prompt,
        "response": answer,
        "response_length_words": word_count,
        "response_length_characters": character_count
    })

    # Conversation history
    conversation_lines.append(f"User: {prompt}\n")
    conversation_lines.append(f"Assistant: {answer}\n")
    conversation_lines.append("-" * 60 + "\n")

# Save responses CSV
pd.DataFrame(responses).to_csv(
    "dataset/llm_responses.csv",
    index=False,
    encoding="utf-8-sig"
)

# Save comparison CSV
pd.DataFrame(comparison_data).to_csv(
    "dataset/comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# Save response length comparison CSV
pd.DataFrame(comparison_data)[
    [
        "prompt_id",
        "prompt",
        "response_length_words",
        "response_length_characters"
    ]
].to_csv(
    "dataset/response_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# Save conversation history TXT
with open(
    "dataset/conversation_history.txt",
    "w",
    encoding="utf-8"
) as file:
    file.writelines(conversation_lines)

print("\nAll files saved successfully!")
print("1. dataset/llm_responses.csv")
print("2. dataset/comparison.csv")
print("3. dataset/response_comparison.csv")
print("4. dataset/conversation_history.txt")