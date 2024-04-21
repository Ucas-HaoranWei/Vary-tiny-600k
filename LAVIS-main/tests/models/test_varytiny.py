"""
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

Integration tests for BLIP2 models.
"""
import argparse
import torch
from lavis.models import load_model, load_model_and_preprocess
from PIL import Image

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







def test_vary_opt125m(image_file):
    model, vis_processors, _ = load_model_and_preprocess(
        name="vary_opt", model_type="inference_opt125m", is_eval=True, device=device
    )

    raw_image = Image.open(image_file).convert("RGB")

    question = 'OCR: '
    
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


    captions = model.generate({"image": image, "prompt": question}, num_captions=1)

    print(captions)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, required=True)
    args = parser.parse_args()


    test_vary_opt125m(args.image_file)

