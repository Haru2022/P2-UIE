import numpy as np
import os
from PIL import Image
from diffusers import UNet2DConditionModel
from diffusers import UNet2DConditionModel,DDIMScheduler,DDPMScheduler
from WaterDecoupleHaruRealPipeline import WaterDecouplePipeline
import torch



unet = UNet2DConditionModel.from_pretrained("HaruCloud9/P2UIE",
                                            subfolder="unet",
                                            torch_dtype=torch.float16,
                                            )

#unet.enable_xformers_memory_efficient_attention()
scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            subfolder="scheduler", 
            timestep_spacing="trailing", # set scheduler timestep spacing to trailing for later inference.
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = WaterDecouplePipeline.from_pretrained("stabilityai/stable-diffusion-2-1", #"stabilityai/stable-diffusion-2-1"
                                                    unet = unet,
                                                    scheduler=scheduler,
                                                    torch_dtype=torch.float16,
                                                    ).to(device)

datasets = ['EUVP','LSUI','haru_1437']

for dataset in datasets:
    test_root = '/media/HDD0/haru/datasets/water-mamba/{}/raw'.format(dataset)
    save_root = '/media/HDD2/haru/Datasets/WaterDecouple/benchmark_result/{}/P2UIE'.format(dataset) 


    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    img_names = os.listdir(test_root)
    img_names = [name for name in img_names if name.endswith('.jpg') or name.endswith('.png')]
    img_names = sorted(img_names)
    img_names = img_names[::-1]

    img_size = 1024
    save_size = 256

    def resize_image(image:Image.Image, size):
        return image.resize((size, size), Image.LANCZOS)

    with torch.no_grad():
        total_inference_time = 0.0  
        measured_batches = 0      
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for img_name in img_names:
            #if not img_name.endswith('.png') or not img_name.endswith('.jpg'):
            #    continue
            image_path = os.path.join(test_root, img_name)
            raw_img = Image.open(image_path).convert("RGB")
            img = resize_image(raw_img, img_size)
            starter.record()
            out = pipeline(
                input_image=img,
                height=img_size,
                width=img_size,
                denoising_steps=1
            )
            ender.record()
            torch.cuda.synchronize()
            inference_time = starter.elapsed_time(ender)
            total_inference_time += inference_time
            measured_batches += 1
            res_pil = out.res_pil
            res_pil = res_pil.resize((save_size, save_size), Image.LANCZOS)
            #res_pil = res_pil.resize(raw_img.size, Image.LANCZOS)
            save_name = os.path.splitext(img_name)[0] + '.png'
            res_pil.save(os.path.join(save_root, save_name))
            print(f"Processed {img_name}, inference time: {inference_time:.2f} ms")

        if measured_batches > 0:
            avg_inference_time = total_inference_time / measured_batches
            print(f"Average inference time for {dataset}: {avg_inference_time:.2f} ms")