import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionInstructPix2PixPipeline
from diffusers.image_processor import VaeImageProcessor
from overrides import override

from .attention import CrossFrameAttnProcessor
from .utils import get_canny_image, get_depth_estimation

class DiffusionPipeline:
    def __init__(self, model_id, device, dtype):
        self.device = device
        self.dtype= dtype
        self.pipeline = self.get_pipeline(model_id)


    def get_pipeline(self, model_id):
        return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)


    def prepare_latent_input_for_unet(self, z_t):
        return torch.cat([z_t] * 2)


    def decode(self, latent, type="pil", do_postprocess=True):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            images = self.vae.decode((1 / self.vae.config.scaling_factor) * latent, return_dict=False)[0].to(device="cpu")
        if do_postprocess:
            images = self.pipeline.image_processor.postprocess(images, output_type=type)
        return images


class DiffusersStableDiffusionPipeline(DiffusionPipeline):
    def __init__(self, model_id, prompt, device="cuda", dtype=torch.float16, do_compile=True):
        super().__init__(model_id, device, dtype)
        self.latent_shape = [self.pipeline.vae.config.latent_channels,
                             self.pipeline.unet.config.sample_size,
                             self.pipeline.unet.config.sample_size]
        self.pixel_space_shape = [3, self.latent_shape[1] * self.pipeline.vae_scale_factor,
                                  self.latent_shape[2] * self.pipeline.vae_scale_factor]

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae

        if do_compile:
            self.unet.compile()
            self.vae.compile()

        self.prediction_type = self.pipeline.scheduler.prediction_type
        with torch.inference_mode():
            self.alphas = torch.sqrt(self.pipeline.scheduler.alphas_cumprod).to(self.device, dtype=dtype)
            self.sigmas = torch.sqrt(1 - self.pipeline.scheduler.alphas_cumprod).to(self.device, dtype=dtype)
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.pipeline.text_encoder.parameters():
            p.requires_grad = False
        
        self.text_embeddings = torch.stack([self.get_text_embeddings(""), self.get_text_embeddings(prompt)], dim=1)
        self.unload_text_encoder()
        
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t)
        timestep = torch.cat([timestep] * 2)
        text_embeddings = torch.cat([self.text_embeddings] * z_t.shape[0])
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        timestep = torch.cat([timestep] * z_t.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * timestep.shape[0]) * e_t + torch.cat([sigma_t] * timestep.shape[0]) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    @torch.no_grad()
    def get_text_embeddings(self, text: str):
        tokens = self.pipeline.tokenizer([text], padding="max_length", max_length=77, truncation=True,
                                return_tensors="pt", return_overflowing_tokens=True).input_ids.to(self.device)
        return self.pipeline.text_encoder(tokens).last_hidden_state.detach()
    
    def unload_text_encoder(self):
        self.pipeline.text_encoder.to("cpu")

    def noise_to_timestep(self, z0, timestep, eps):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z0 + sigma_t * eps
        return z_t
    



class Teacher(DiffusersStableDiffusionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward_operator(self, x):
        raise NotImplementedError
    
    def encode(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_enc = self.pipeline.vae.encode(x).latent_dist.mean * self.pipeline.vae.config.scaling_factor
        return x_enc
    



class InstructPixToPixTeacher(Teacher):
    def __init__(self, gt_images, editing_prompt, image_guidance_scale, **kwargs):
        super().__init__(**kwargs)

        self.gt_images = [image.resize(self.pixel_space_shape[1:]) for image in gt_images]

        self.pipeline.text_encoder.cuda()
        self.editing_prompt_embeds = self.pipeline._encode_prompt(
                editing_prompt,
                self.device,
                1,
                True,
                negative_prompt=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
            )
        self.unload_text_encoder()

        self.source_image_latents = self.prepare_image_latents(self.gt_images)
        self.image_guidance_scale = image_guidance_scale # needs to be >=1. Higher image guidance scale encourages generated images that are closely
                                        #linked to the source `image`, usually at the expense of lower image quality

    @override
    def get_pipeline(self, model_id):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
                                                                     torch_dtype=self.dtype,
                                                                     safety_checker = None,
                                                                     requires_safety_checker = False).to(self.device)
        return pipe
    
    def set_attn_processor(self, processor_name):
        if processor_name == "cross_frame":
            self.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
        else:
            raise ValueError(f"Unknown processor name: {processor_name}")

    def prepare_image_latents(self, source_images):
        source_images = self.pipeline.image_processor.preprocess(source_images).to(device=self.device, dtype=self.dtype)
        source_image_latents = self.vae.encode(source_images).latent_dist.mode()
        source_image_latents = torch.cat([source_image_latents], dim=0)
        uncond_image_latents = torch.zeros_like(source_image_latents)
        source_image_latents = torch.cat([source_image_latents, source_image_latents, uncond_image_latents], dim=0)
        return source_image_latents


    @override
    def prepare_latent_input_for_unet(self, z_t):
        z_t = torch.cat([z_t] * 3)
        source_images = self.source_image_latents 
        return torch.cat([z_t, source_images], dim=1)


    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t)
        timestep = torch.cat([timestep] * 2)
        text_embeddings = torch.cat([self.editing_prompt_embeds.unsqueeze(0)] * z_t.shape[0])
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # predict the noise residual
            noise_pred = self.unet(
                latent_input,
                timestep[0],
                encoder_hidden_states=embedd,
                added_cond_kwargs=None,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

        pred_z0 = (z_t - sigma_t * noise_pred) / alpha_t
        return noise_pred, pred_z0
    

    @override
    def forward_operator(self, x):
        return x
    
    
class ControlNetTeacher(Teacher):
    def __init__(self, gt_images, controlnet_id, **kwargs):
        self.controlnet_id = controlnet_id
        super().__init__(**kwargs)

        self.gt_images = [image.resize(self.pixel_space_shape[1:]) for image in gt_images] # for plot input
        cond_image = self.forward_operator(self.gt_images)

        self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        self.cond_image = self.prepare_cond_image(cond_image)

    @override
    def get_pipeline(self, model_id):
        controlnet = ControlNetModel.from_pretrained(self.controlnet_id, torch_dtype=self.dtype).to(self.device)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=self.dtype).to(self.device)
        return pipe
    
    def set_attn_processor(self, processor_name):
        if processor_name == "cross_frame":
            self.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
        else:
            raise ValueError(f"Unknown processor name: {processor_name}")

    @torch.no_grad()
    def prepare_cond_image(self, cond_image):
        height, width = cond_image[0].height, cond_image[0].width
        cond_image = self.control_image_processor.preprocess(cond_image, height=height, width=width).to(dtype=self.dtype)
        cond_image = cond_image.to(device=self.device, dtype=self.dtype)

        return cond_image

    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t)
        timestep = torch.cat([timestep] * 2)
        text_embeddings = torch.cat([self.text_embeddings] * z_t.shape[0])
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        timestep = torch.cat([timestep] * z_t.shape[0])
        controlnet_cond = torch.cat([self.cond_image] * 2)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                latent_input,
                timestep,
                encoder_hidden_states=embedd,
                controlnet_cond=controlnet_cond,
                conditioning_scale=1.0,
                guess_mode=False,
                return_dict=False,
            )

            e_t = self.unet(latent_input, timestep, embedd, down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample).sample
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    

class DepthTeacher(ControlNetTeacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def forward_operator(self, x):
        """
        @param x: a torch tensor in image space of shape [b, 3, H, W]
        """
        return get_depth_estimation(x)
    

class CannyTeacher(ControlNetTeacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def forward_operator(self, x):
        """
        @param x: a torch tensor in image space of shape [b, 3, H, W]
        """
        return get_canny_image(x)


def get_teacher(teacher_name, **kwargs):
    if teacher_name == "pix2pix":
        return InstructPixToPixTeacher(gt_images=kwargs["gt_images"],
                                       prompt=kwargs["prompt"],
                                        editing_prompt=kwargs["editing_prompt"],
                                        image_guidance_scale=kwargs["image_guidance_scale"],
                                       model_id="timbrooks/instruct-pix2pix",
                                       device=kwargs["device"],
                                       dtype=torch.float16,
                                       do_compile=kwargs["do_compile"])
        
    if teacher_name == "canny":
        return CannyTeacher(gt_images=kwargs["gt_images"],
                            prompt=kwargs["prompt"],
                            controlnet_id="lllyasviel/sd-controlnet-canny",
                            model_id="runwayml/stable-diffusion-v1-5",
                            device=kwargs["device"],
                            dtype=torch.float16,
                            do_compile=kwargs["do_compile"])
        
    if teacher_name == "depth":
        return DepthTeacher(gt_images=kwargs["gt_images"],
                            prompt=kwargs["prompt"],
                            controlnet_id="lllyasviel/sd-controlnet-depth",
                            model_id="runwayml/stable-diffusion-v1-5",
                            device=kwargs["device"],
                            dtype=torch.float16,
                            do_compile=kwargs["do_compile"])
        
    raise ValueError("Wrong teacher_name: {}".format(teacher_name))