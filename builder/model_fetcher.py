'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''


from huggingface_hub import login
login("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")


def download_model():
    '''
    Downloads the model from the URL passed in.
    '''
    from huggingface_hub import login
    login("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")

    from huggingface_hub import snapshot_download
    models = ['black-forest-labs/FLUX.1-dev', "sayakpaul/flux.1-dev-nf4-pkg"]
    for model in models:
        snapshot_download(repo_id=model, local_dir=f"/workspace/{model.split('/')[1]}")

download_model()