import base64

with open('../model_29.ckpt.data-00000-of-00001', 'rb') as f:
    encoded_string = base64.b64encode(f.read())

with open('serialized.py', 'w') as f:
    f.write(f'model_string={encoded_string}')

with open('../model_29.ckpt.index', 'rb') as f:
    encoded_string_index = base64.b64encode(f.read())

with open('serialized_index.py', 'w') as f:
    f.write(f'model_string={encoded_string_index}')

with open('../model_29.ckpt.meta', 'rb') as f:
    encoded_string_meta = base64.b64encode(f.read())

with open('serialized_meta.py', 'w') as f:
    f.write(f'model_string={encoded_string_meta}')