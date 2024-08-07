import os
from pathlib import Path
import requests
import cv2

base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"


def get_model(model_name):
    model_folder = Path(os.path.dirname(os.path.realpath(__file__)) + "/models/")
    model_weight =  os.path.join(str(model_folder), f"{model_name}.pt")
    if not os.path.isfile(model_weight):
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        print("Downloading the model...")
        model_url = f"{base_url}{model_name}.pt"
        response = requests.get(model_url)
        with open(model_weight, 'wb') as f:
            f.write(response.content)

    config_folder = os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            "sam_2",
                            "sam2_configs"
    )

    config_list = {"sam2_hiera_small": "sam2_hiera_s.yaml",
                    "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
                    "sam2_hiera_large": "sam2_hiera_l.yaml",
                    "sam2_hiera_tiny": "sam2_hiera_t.yaml"}

    cfg_name = config_list[model_name]

    return model_weight, config_folder, cfg_name

def resize_mask(mask, h_orig, w_orig):
    mask = cv2.resize(
                        mask,
                        (w_orig, h_orig),
                        interpolation = cv2.INTER_NEAREST
        )
    return mask