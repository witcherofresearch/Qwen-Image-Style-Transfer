import gradio as gr
import numpy as np
import random
import torch
#import spaces

from PIL import Image
#from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

import os
from huggingface_hub import  hf_hub_download

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
from model import CSD_CLIP, convert_state_dict
from PIL import Image
import os
import json
import re
import glob
import argparse

# 初始化模型
model = CSD_CLIP("vit_large", "default", model_path="/gemini/platform/public/aigc/cv_banc/zsw/style_transfer_ckpt/CSD_SCORE/ViT-L-14.pt")

# 加载模型
model_path = "/gemini/platform/public/aigc/cv_banc/zsw/style_transfer_ckpt/CSD_SCORE/checkpoint.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
state_dict = convert_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(state_dict, strict=False)
model = model.cuda()

# 图像预处理
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
preprocess = transforms.Compose([
    transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


#speedup = hf_hub_download(repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning", filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")



##diffusers

import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests

qweneditdir='/gemini/platform/public/aigc/aigc_image_datasets/huggingface/models/Qwen/Qwen-Image-Edit-2509'

pipe = QwenImageEditPlusPipeline.from_pretrained(qweneditdir, torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipe.to('cuda')
pipe.set_progress_bar_config(disable=None)



lora_model='/gemini/platform/public/aigc/cv_banc/zsw/qwenimageedit/DiffSynth-Studio/models/train/diffusers_Qwen-Image-Edit-2509-Style-Transfer-V1.safetensors'
print("loading lora")

pipe.load_lora_weights(
    lora_model,adapter_name='style'
)

dmd_lora='/gemini/platform/public/aigc/cv_banc/zsw/qwenimageedit/DiffSynth-Studio/models/train/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors'

pipe.load_lora_weights(
    dmd_lora,adapter_name='dmd'
)

pipe.set_adapters(["style", "dmd",], adapter_weights=[1.0, 1.0])
#pipe.fuse_lora(adapter_names=["style", "dmd"], lora_scale=1.0)
#pipe.unload_lora_weights()
#save_root='/gemini/platform/public/aigc/cv_banc/zsw/qwenimageedit/DiffSynth-Studio/models/train/fuse_lora_qwenstyle-qwenedit2509'
#pipe.save_pretrained(save_root)


dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"




MAX_SEED = np.iinfo(np.int32).max


#@spaces.GPU
def infer(
    content_ref,
    style_ref,
    prompt,
    seed=123,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    minedge=1024,
    progress=gr.Progress(track_tqdm=True),
    
):
    
    

    

    content_ref=Image.fromarray(content_ref)
    style_ref=Image.fromarray(style_ref)
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    
    
    
    
    
    
    w,h=content_ref.size



    #minedge=1024
    if w>h:
        r=w/h
        h=minedge
        w=int(h*r)-int(h*r)%16
        
    else:
        r=h/w
        w=minedge
        h=int(w*r)-int(w*r)%16


    
    print(f"Calling pipeline with prompt: '{prompt}'")
    
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}, Size: {w}x{h}")
    
    images = [
        content_ref.resize((w, h)),
        style_ref.resize((minedge, minedge)) ,
    ]
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_guidance_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        "width": w,
        "height": h, 
    }
    with torch.inference_mode():
        image = pipe(**inputs)
    image = image.images[0]
    

    
    
    # style image
    output = preprocess(image).unsqueeze(0).to("cuda")
    _, content_output, style_output = model(output)

    # another style image
    image1 = preprocess(style_ref).unsqueeze(0).to("cuda")
    _, content_output1, style_output1 = model(image1)

    sim = style_output@style_output1.T
    print(sim)


    return image, seed,sim.item()

# --- Examples and UI Layout ---
examples = []
'''
css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""
'''


_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">QwenStyle V1</h1>
    
</div>


<p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='https://openreview.net/forum?id=Cgb7JpOA5Q&referrer=%5Bthe%20profile%20of%20Shiwen%20Zhang%5D(%2Fprofile%3Fid%3D~Shiwen_Zhang1)' target='_blank'>QwenStyle: Content-Preserving Style Transfer with Qwen-Image-Edit</a> | Codes: <a href='https://github.com/witcherofresearch/Qwen-Image-Style-Transfer' target='_blank'>GitHub</a></p>
'''  


#with gr.Blocks(css=css) as demo:
with gr.Blocks() as demo:

    with gr.Column(elem_id="col-container"):
        gr.HTML('<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">')
        gr.Markdown(_HEADER_)
        gr.Markdown("This is a demo of QwenStyle v1, the first Content-Preserving Style Transfer Lora on Qwen-Image-Edit-2509.")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    content_ref = gr.Image(label="content ref", type="numpy", )
                    style_ref = gr.Image(label="style ref", type="numpy", )
                    print(f"type(content_ref)={type(content_ref)}")
                    
                #input_images = gr.Gallery(label="Input Images", show_label=False, type="pil", interactive=True)
            
            with gr.Column():
                result = gr.Image(label="Result", show_label=True, type="pil")
                sim = gr.Number(
                        label="Style Similarity (sim)",
                        value=0.0,
                        precision=6,  # 保留6位小数
                        interactive=False  # 不可编辑
                    )

            #result = gr.Gallery(label="Result", show_label=True, type="pil")
        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    value='Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.',
                    show_label=True,
                    placeholder='Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.',
                    container=True,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=True):
            # Negative prompt UI element is removed here

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=123,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=False)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="CFG should be 1.0",
                    minimum=0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps should be 4",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                )
                
                minedge = gr.Slider(
                    label="Min Edge of the generated image",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=1024,
                )
        '''
        with gr.Row(), gr.Column():
            gr.Markdown("## Examples")
            gr.Markdown("changing the minedge could lead to different style similarity.")
            default_prompt='Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'
            gr.Examples(examples=[
                ['./qwenstyleref/pulpfiction_2.jpg','./qwenstyleref/styleref=6_style_ref.png',default_prompt],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/110.png',default_prompt],
                ['./qwenstyleref/romanholiday_1.jpg','./qwenstyleref/s0099____1113_01_query_1_img_000146_1682705733350_08158389675901344.jpg.jpg',default_prompt],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/125.png',default_prompt],
                ['./qwenstyleref/fallenangle.jpg','./qwenstyleref/styleref=s0038.png',default_prompt],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/styleref=s0572.png',default_prompt],
                ['./qwenstyleref/startrooper1.jpg','./qwenstyleref/david-face-760x985.jpg','Style Transfer Figure  1 into marble material.'],
                ['./qwenstyleref/possession.png','./qwenstyleref/s0026____0907_01_query_0_img_000194_1682674358294_041656249089406583.jpeg.jpg',default_prompt],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/Jotarokujo.webp',default_prompt],
                ['./qwenstyleref/wallstreet1.jpg','./qwenstyleref/034.png',default_prompt],
                ['./qwenstyleref/bird.jpeg','./qwenstyleref/styleref=s0539.png',default_prompt],
                

                #['/gemini/platform/public/aigc/cv_banc/zsw/qwenimageedit/DiffSynth-Studio/output/testligtninglora/showlabseedstyle30k/qwenedit+showlab+seed+pretrain+style30k_eachstyleref=1comfyui_initfrom92000/step-30000.safetensors/_minedge=1024_styleref=1024_edit_image_auto_resize=False/03d90e38fc3b2016dafaa0c2df57019d/styleref=0_content_ref.png','/gemini/platform/public/aigc/cv_banc/zsw/qwenimageedit/DiffSynth-Studio/output/testligtninglora/showlabseedstyle30k/qwenedit+showlab+seed+pretrain+style30k_eachstyleref=1comfyui_initfrom92000/step-30000.safetensors/_minedge=1024_styleref=1024_edit_image_auto_resize=False/03d90e38fc3b2016dafaa0c2df57019d/styleref=8_style_ref.png',default_prompt],
                #['/gemini/platform/public/aigc/cv_banc/zsw/qwenimageedit/DiffSynth-Studio/output/testligtninglora/showlabseedstyle30k/qwenedit+showlab+seed+pretrain+style30k_eachstyleref=1comfyui_initfrom92000/step-30000.safetensors/_minedge=1024_styleref=1024_edit_image_auto_resize=False/maxresdefault/styleref=0_content_ref.png','/gemini/platform/public/aigc/cv_banc/zsw/qwenimageedit/DiffSynth-Studio/output/testligtninglora/showlabseedstyle30k/qwenedit+showlab+seed+pretrain+style30k_eachstyleref=1comfyui_initfrom92000/step-30000.safetensors/_minedge=1024_styleref=1024_edit_image_auto_resize=False/maxresdefault/styleref=11_style_ref.png',default_prompt]
                ],
                inputs=[content_ref,style_ref, prompt], 
                outputs=[result, seed], 
                fn=infer, 
                cache_examples=False
                )        
                
                
                
                

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)
        '''

    gr.on(
        triggers=[run_button.click],
        fn=infer,
        inputs=[
            content_ref,
            style_ref,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            minedge,
            
        ],
        outputs=[result, seed,sim],
    )

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
