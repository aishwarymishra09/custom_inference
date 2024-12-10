'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import shutil
import requests
import argparse
from pathlib import Path
from urllib.parse import urlparse
from transformers import T5EncoderModel
from diffusers import StableDiffusionPipeline
from diffusers import FluxPipeline, FluxTransformer2DModel
from huggingface_hub import login
login("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")
SAFETY_MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_CACHE_DIR = "diffusers-cache"
ckpt_4bit_id = "sayakpaul/flux.1-dev-nf4-pkg"

def download_model(model_url: str):
    '''
    Downloads the model from the URL passed in.
    '''
    model_cache_path = Path(MODEL_CACHE_DIR)
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)

    # Check if the URL is from huggingface.co, if so, grab the model repo id.
    parsed_url = urlparse(model_url)
    if parsed_url.netloc == "huggingface.co":
        model_id = f"{parsed_url.path.strip('/')}"
    else:
        downloaded_model = requests.get(model_url, stream=True, timeout=600)
        with open(model_cache_path / "model.zip", "wb") as f:
            for chunk in downloaded_model.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    # StableDiffusionSafetyChecker.from_pretrained(
    #     SAFETY_MODEL_ID,
    #     cache_dir=model_cache_path,
    # )

    FluxPipeline.from_pretrained(
        model_id,
        cache_dir=model_cache_path,
    )
    T5EncoderModel.from_pretrained(
        ckpt_4bit_id,
        subfolder="text_encoder_2",
    )


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model_url", type=str,
    default="https://huggingface.co/stabilityai/stable-diffusion-2-1",
    help="URL of the model to download."
)

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_url)
