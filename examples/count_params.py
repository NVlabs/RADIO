# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import time
from tqdm import tqdm

import torch


from common import load_model

MODELS = [
    ('OpenAI CLIP', 'open_clip,ViT-L-14-336,openai', 336, 16),
    ('OpenCLIP', 'open_clip,ViT-H-14,laion2b_s32b_b79k', 224, 16),
    ('DFN CLIP', 'open_clip,ViT-H-14-378-quickgelu,dfn5b', 378, 16),
    ('SigLIP', 'open_clip,ViT-SO400M-14-SigLIP-384,webli', 384, 16),
    ('MetaCLIP', 'open_clip,ViT-H-14-quickgelu,metaclip_fullcc', 224, 16),

    ('DINOv2-g-reg', 'dinov2_vitg14', 224, 16),

    ('SAM-H', 'sam,/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/model_zoo/sam/sam_vit_h_4b8939.pth', 1024, 4),

    ('RADIO', 'radio_v2.1', 432, 16),

    ('InternViT-6b-224', 'InternViT-6B-224px', 224, 8),
    ('InternViT-6B-448-1.2', 'InternViT-6B-448px-V1-2', 448, 8),
]

class xyz_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        ff = self.model(tensor)
        return ff

@torch.inference_mode()
def main(rank: int = 0, world_size: int = 1):
    prms = []
    for name, version, resolution, batch_size in MODELS:
        print(f'Loading "{name}"...')
        model, preproc, _ = load_model(version)
        if hasattr(model, 'vision_encoder'):
            model = model.vision_encoder
        print(f'Done')

        if version.startswith('dinov2'):
            model = xyz_model(model)

        num_params = sum(p.numel() for p in model.parameters() if p is not None and p.requires_grad)

        num_params_m = num_params / 1_000_000

        print(f'Num Params: {num_params_m:.1f}M')

        print(f'Calculating throughput...')
        model.cuda().eval()
        preproc.cuda().eval()
        buff = torch.empty(batch_size, 3, resolution, resolution, dtype=torch.float32, device='cuda')
        pt_buff = preproc(buff)
        throughput = 0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            # First one is free
            model(pt_buff)
            torch.cuda.synchronize()
            start_time = time.time()
            NUM_BATCHES = 100
            for _ in tqdm(range(NUM_BATCHES)):
                model(pt_buff)
            torch.cuda.synchronize()
            end_time = time.time()

        throughput = (NUM_BATCHES * buff.shape[0]) / (end_time - start_time)
        print(f'Done. {throughput:.2f} im/sec')

        # # Uncomment to profile model with TensorRT
        # model.float()
        # onnx_file_path = '/tmp/trt_model.onnx'
        # torch.onnx.export(
        #     model,
        #     buff,
        #     onnx_file_path,
        #     input_names=['input'],
        #     output_names=['output'],
        #     export_params=True,
        #     opset_version=19,
        # )
        # os.system(f'trtexec --onnx={onnx_file_path} --fp16 --allowGPUFallback --workspace=300000000')
        # print('\n\n\n\n\n')

        prms.append((name, num_params_m, throughput))

    with open('model_parameters.csv', 'w') as fd:
        fd.write('Name,Parameters (M),Throughput\n')
        for name, ct, throughput in prms:
            fd.write(f'{name},{ct:.1f},{throughput:.2f}\n')

if __name__ == '__main__':
    main()
