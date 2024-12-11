INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'id': {
        'type': str,
        'required': True,
        'default': None
    },
    'request_id': {
        'type': str,
        'required': True,

    },
    'lora': {
        'type': str,
        'required': False,
        'default': None
    }}
