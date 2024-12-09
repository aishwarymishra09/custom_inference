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
    'width': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768]
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
