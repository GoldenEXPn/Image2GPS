from pathlib import Path
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import login 

def load_hf_token(token_path: str = "huggingface_token") -> str:
    token = Path(token_path).read_text().strip()
    if not token:
        raise ValueError("Hugging Face token file is empty.")
    return token

hf_token = load_hf_token("huggingface_token")
login(token=hf_token)


video_data = load_dataset("tianyi-in-the-bush/penncampus_image2gps")

video_360_data = load_dataset("aaron-jiang/penncampus_image2gps")

merged_dataset = {
    "train": concatenate_datasets([video_data['train'], video_360_data['train']]),
    "test": concatenate_datasets([video_data['test'], video_360_data['test']]),
}

merged_dataset = DatasetDict(merged_dataset)
merged_dataset.push_to_hub("aaron-jiang/penncampus_image2gps_merged")