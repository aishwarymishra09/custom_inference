''' infer.py for runpod worker '''
import os

import boto3
import argparse
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from runpod.serverless.utils import rp_download, rp_cleanup
from io import BytesIO
from rp_schema import INPUT_SCHEMA
from flux_inf_quant import inference_sample
from utils.logger import logger


def encode_key(text, shift):
    """
    Encode an alphanumeric key by shifting characters.

    :param text: The alphanumeric key to encode.
    :param shift: The number of positions to shift each character.
    :return: Encoded key as a string.
    """
    encoded = []
    for char in text:
        if char.isalpha():
            # Shift alphabetic characters (wrap within 'a-z' or 'A-Z')
            base = ord('a') if char.islower() else ord('A')
            encoded.append(chr((ord(char) - base + shift) % 26 + base))
        elif char.isdigit():
            # Shift numeric characters (wrap within '0-9')
            encoded.append(chr((ord(char) - ord('0') + shift) % 10 + ord('0')))
        else:
            # Keep other characters unchanged
            encoded.append(char)
    return ''.join(encoded)
ACCESS_KEY_ID = encode_key("EOMECW6RXDRLERXEEI1Q", -4)
SECRET_ACCESS_KEY = encode_key("Wwcb3I78eyTDReRCOnTvQk5qToGAsncZTE1rXITy", -4)
BUCKET_NAME = "rekogniz-training-data"

def save_image(image, path):
    """Uploads an image to an S3 bucket"""
    try:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
        out_img = BytesIO()
        image.save(out_img, format='png')
        out_img.seek(0)
        s3.upload_fileobj(out_img, BUCKET_NAME, path,
                          ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read', 'ContentDisposition': 'inline'})
        REMOTE_IMAGE_FILE = "https://rekogniz-training-data.s3.ap-south-1.amazonaws.com/infernce-rekogniz/{}/{}/sample_{}.png"

    except Exception as e:
        return e
    return True


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    all_images = inference_sample(validated_input['id'], validated_input['request_id'], validated_input['lora'],
                                 validated_input['prompt'])

    job_output = []
    img_remote_path = "infernce-rekogniz/"+ f"{job['id']}/" + validated_input['id'] + f"/{validated_input['request_id']}" + "/sample_{}.png"
    REMOTE_IMAGE_FILE = f"https://rekogniz-training-data.s3.ap-south-1.amazonaws.com/infernce-rekogniz/{job['id']}/{validated_input['id']}/{validated_input['request_id']}/" +"sample_{}.png"

    for i, im in enumerate(all_images):
        save_image(im, img_remote_path.format(i + 1))

        file_name = img_remote_path.format(i)
        save_image(im, file_name)

        job_output.append(REMOTE_IMAGE_FILE.format(i))

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output


if __name__ == "__main__":
    logger.info("starting inference ...")
    runpod.serverless.start({"handler": run})
