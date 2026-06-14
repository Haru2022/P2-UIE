from typing import Any, Dict, Union

import torch.nn as nn
import torch
#from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
#from models.unet2dconditionwater import UNet2DConditionModel
#from ..models.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import BaseOutput
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer,CLIPTextModel
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
#from util.physical_encoder import physical_encoder
import warnings
#from util.WaterPhysicalLosses_v3 import disp_to_depth, depth_to_disp

import cv2

class WaterDecouplePipelineOutput(BaseOutput):
    res_np: np.ndarray = None
    res_pil: Image.Image = None
    #res_tensor: torch.Tensor

class WaterDecouplePipeline(DiffusionPipeline):
    
    latent_scale_factor = 0.18215
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
                 tokenizer:CLIPTokenizer,
                 text_encoder:CLIPTextModel,
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.img_embed = None 
        self.task_embed = None
        self.pre_latents = None
        self.pre_latents_ratio = 0.1
        
    # Apply VAE Encoder to image   
    def encode_RGB(self, image):
        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        latent, _ = torch.chunk(moments, 2, dim=1)
        latent = latent * self.latent_scale_factor
        return latent

    # Apply VAE Decoder to latent
    def decode_RGB(self, latent):
        latent = latent / self.latent_scale_factor
        z = self.vae.post_quant_conv(latent)
        image = self.vae.decoder(z)
        return image
    
        
    def task_prompt_encode(self):
        """
        Encode text embedding for empty prompt
        """

        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            #max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        if isinstance(self.unet, UNet2DConditionModel):
            self.task_embed = self.text_encoder(text_input_ids,return_dict=False)[1].to(self.dtype)
        else:
            raise NotImplementedError
            self.task_embed = self.text_encoder(text_input_ids)[1].to(self.dtype)
            
    def input_preprocess(self, input_image:Union[Image.Image, np.ndarray], height=None, width=None):
        assert isinstance(input_image, (Image.Image, np.ndarray)), "input_image should be PIL Image or np.ndarray"
        if isinstance(input_image, np.ndarray): # only for sequential inference, no resize operation
            out = torch.from_numpy(input_image.copy()).to(self.dtype).to(self.device)
            if out.dim() == 2:
                out = out.unsqueeze(0).repeat(3,1,1)
            else:
                out = out.permute((2,0,1)).contiguous()
            out = out*2.0 - 1.0 # [0, 1] -> [-1, 1]
        elif isinstance(input_image, Image.Image):
            input_width, input_height = input_image.size
            if height != input_height or width != input_width:
                print("resize")
                input_image = input_image.resize((width, height))
                
            if input_image.mode == "RGB":
                image = np.array(input_image)
                out = np.transpose(image, (2, 0, 1))
                out = out / 255.0 * 2.0 - 1.0 # [0, 255] -> [-1, 1]
            else:
                image = np.array(input_image)
                image = (image[:, :, np.newaxis]/65535.0) *2.0 -1.0 # [0, 65535] -> [-1, 1]
                out = image.repeat(3,axis=2).transpose(2, 0, 1)
            out = torch.from_numpy(out).to(self.dtype).to(self.device)
        out = out.unsqueeze(0)
        return out
        
    def output_postprocess(self, output:torch.Tensor):
        output = output.clamp(-1.0, 1.0)
        output = (output+1.0) / 2.0
        output_np = output.cpu().numpy().transpose(1, 2, 0)
        if output_np.shape[2] == 1:
            #print(output_np.shape)
            #output_np = np.tile(output_np, (1, 1, 3))
            output_pil = Image.fromarray((output_np * 65535).astype(np.uint16),mode='I;16')
        else:
            output_pil = Image.fromarray((output_np * 255).astype(np.uint8))
        return output_np, output_pil
                

    @torch.no_grad()
    def __call__(self,
                 input_image: Union[Image.Image, np.ndarray],
                 denoising_steps: int = 1,
                 height:int = None,
                 width:int = None,
                 inner_temp_consistency:bool = False,
                 ) -> WaterDecouplePipelineOutput:
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        self.img_embed = None
        rgb = self.input_preprocess(input_image, height, width)
        rgb_latent = self.encode_RGB(rgb)
        
        conditonal_latents = rgb_latent

        self.scheduler.set_timesteps(denoising_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        

        latent = torch.zeros(rgb_latent.shape, device=device, dtype=self.dtype)      
        if inner_temp_consistency and self.pre_latents is not None:
            latent = self.pre_latents_ratio*self.pre_latents+(1-self.pre_latents_ratio)*latent  

        self.task_prompt_encode()
        batch_task_embed = self.task_embed.repeat(
            (1, 1, 1))
        
        
        # Denoising loop

        iterable = enumerate(timesteps)
            
        for i, t in iterable:
            
            unet_input = torch.cat(
                [conditonal_latents, latent], dim=1
            )  # this order is important
            
            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_task_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_step = self.scheduler.step(
                noise_pred, t, latent
            )
            
            
            latent = scheduler_step.prev_sample

            # add
            #if i == denoising_steps-1:
            #    latent = scheduler_step.pred_original_sample
        #print(latent.shape)
        self.pre_latents = latent
        torch.cuda.empty_cache()
        air_res = self.decode_RGB(latent).squeeze(0)
        air_np, air_pil = self.output_postprocess(air_res)
        
        return WaterDecouplePipelineOutput(res_nps=air_np,
                                           res_pil=air_pil)
        
            
            
        
        
        
        
        
        
        
        
        
    