# -*- coding: utf-8 -*-
# @Time    : 6/24/21 12:50 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : demo.py


# Notes:
# 0. Build with torch@8a80cee2f3d, torchvision@7d077f1312; ran on Mac M1
#
# 1. `requirements.txt` requires older Python, so I manually installed the
# dependencies I needed:
# ```
# pip install wget timm==0.4.5
# ```
# 
# 2. must run `src/demo.py` by changing directory first...
# ```
# cd src/models
# python ../demo.py
# ```
# 
# 3. aoti output had some non-obvious mismatch from the original output need (this
#    happens consistently but without different tolerances since the runs seem to
#    be somewhat indeterministic)
# - `r_tol = 6e-3` (default `1e-5`) or
# - `a_tol = 4e-6` (default `1e-8`)
#
# Perf:
#     original module took 0.2246 seconds
#     exported model took 0.2209 seconds, speed-up: 1.0165814223819687
#     aoti-ed model took 0.1822 seconds, speed-up: 1.232303254841662


import os
import time
import torch
from models import ASTModel
# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'
# assume each input spectrogram has 100 time frames
input_tdim = 100
# assume the task has 527 classes
label_dim = 527
# create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
test_input = torch.rand([10, input_tdim, 128])
# create an AST model
original_model = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=True)


def bench(fn):
    # Warm it up
    for i in range(3):
        fn()

    # Time it
    start = time.perf_counter()
    res = fn()
    end = time.perf_counter()
    return res, end - start


with torch.no_grad():
    # Run original model
    original_output, original_dur = bench(lambda: original_model(test_input))
    print(f"original module took {original_dur:.4f} seconds")


    # Export
    batch = torch.export.Dim("batch", min=1, max=1024)
    example_inputs = (test_input,)
    dynamic_shapes = { "x": { 0 : batch } }
    exported = torch.export.export(original_model, example_inputs, dynamic_shapes=dynamic_shapes)
    #print(exported) # visualize the exported model


    # Run exported model
    exported_model = exported.module()
    exported_output, exported_dur = bench(lambda: exported_model(test_input))
    exported_output_match = torch.allclose(original_output, exported_output)
    print(f"exported model took {exported_dur:.4f} seconds, speed-up: {original_dur / exported_dur}")
    print(f"{exported_output_match=}")


    # Run AOTI compile and package
    aoti_output_path = torch._inductor.aoti_compile_and_package(
        exported,
        example_inputs,
        package_path=os.path.join(os.getcwd(), "model.pt2"),
    )


    # Load and run AOTI model
    aoti_model = torch._inductor.aoti_load_package(aoti_output_path)
    aoti_output, aoti_dur = bench(lambda: aoti_model(test_input))
    aoti_output_match = torch.allclose(original_output, aoti_output)
    print(f"aoti-ed model took {aoti_dur:.4f} seconds, speed-up: {original_dur / aoti_dur}")
    print(f"{aoti_output_match=}")