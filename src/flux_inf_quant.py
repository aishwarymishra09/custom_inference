import argparse
import time
import boto3
import pickle
from io import BytesIO
from diffusers import FluxPipeline, FluxTransformer2DModel
from constants.properties import ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME
from transformers import T5EncoderModel
import torch
import gc
from huggingface_hub import login
login("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")

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


def save_object_to_file(obj, filename="sdxl.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_object_from_file(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj





def inference_sample(id, request_id, lora_weight, prompt, ckpt_4bit_id="sayakpaul/flux.1-dev-nf4-pkg",
                     ckpt_id="black-forest-labs/FLUX.1-dev"):
    """This function is used to create sample using flux model"""
    flush()
    print(lora_weight)
    cc = time.time()
    text_encoder_2_4bit = T5EncoderModel.from_pretrained(
        ckpt_4bit_id,
        subfolder="text_encoder_2",
    )
    tc = time.time()
    print(f"time taken by text_enc:{tc - cc}")
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder_2=text_encoder_2_4bit,
        transformer=None,
        vae=None,
        torch_dtype=torch.float16,
    )
    pipeline.enable_model_cpu_offload()
    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=[prompt] * 3, prompt_2=None, max_sequence_length=256
        )
    pipeline = pipeline.to("cpu")
    del pipeline
    pc = time.time()
    print(f"time taken by pipeline:{pc - tc}")
    flush()
    transformer_4bit = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer")
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer_4bit,
        torch_dtype=torch.float16,
    )
    p2c = time.time()
    print(f"time taken by pipeline2:{p2c - tc}")
    # save_object_to_file(pipeline, filename="flux.pkl")
    # pipeline = load_object_from_file("flux.pkl")

    pipeline.load_lora_weights(lora_weight)
    pipeline.enable_model_cpu_offload()

    print("Running denoising.")
    height, width = 512, 768

    all_images = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=50,
        guidance_scale=5.5,
        height=height,
        width=width,
        output_type="pil",
    ).images



    # image[0].save("hh.png")


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--lora', help='target string for query', default=None,
                        type=str)

    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument('--negative_prompt', help='negative prompt to generate', default=None,
                        type=str)
    parser.add_argument('--id', help='target string for query', default=None,
                        type=str)
    parser.add_argument('--request_id', help='target string for query', default=None,
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    c = time.time()
    args = parse_args()
    inference_sample(args.id, args.request_id, args.lora, args.prompt)
    print(f"Done in {time.time() - c}")
