import os
from feature_extraction import mediapipe_prompt_extraction

def get_dir_from_root(path: str) -> str: 
    return os.path.join(os.path.abspath(".."), path)

mediapipe_prompt_extraction(get_dir_from_root('data/demo/Humans/280.jpg'))
# mediapipe_prompt_extraction('data/demo/280.jpeg')