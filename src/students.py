

import torch
from PIL import Image
from overrides import override
from einops import repeat

from seva.utils import load_model
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser, NoCFG, append_dims, to_d
from seva.eval import unload_model


class SevaPipeline:
    def __init__(self, device, value_dict, do_compile=True, use_grad_checkpointing=False):
        self.device = device
        self.model = SGMWrapper(load_model(weight_name="modelv1.1.safetensors", device="cpu", verbose=True)).to(device=self.device)
        self.model.use_grad_checkpointing = use_grad_checkpointing
        # loading to cpu to save memory
        self.ae = AutoEncoder(chunk_size=1).to(device=self.device)
        self.ae.module.eval().requires_grad_(False)


        if do_compile:
            self.model.compile()
            self.ae.compile()

        self.conditioner = CLIPConditioner().to("cpu")
        self.discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(discretization=self.discretization, num_idx=1000, device=device)
        self.guider = NoCFG()

        self.value_dict = value_dict
        self.version_dict = self.value_dict["version_dict"]
        self.n_input_frames = value_dict["cond_frames_mask"].sum().item()
        self.n_padding = self.version_dict["T"] - value_dict["num_imgs_no_padding"]

        self.latent_shape = [self.version_dict["T"],self.version_dict["C"],
                self.version_dict["H"] // self.version_dict["f"], self.version_dict["W"] // self.version_dict["f"]] # [T, 4, 72, 72] 

        self.c, self.uc, self.additional_model_inputs, self.additional_sampler_inputs = self.prepare_model_inputs()

        self.model.train()
        self.set_model_requires_grad()

    def set_model_requires_grad(self):
        self.model.requires_grad_(False)

    def prepare_model_inputs(self):
        """
        adapted from seva's eval.py do_sample function.
        """
        
        imgs = self.value_dict["cond_frames"].to(device=self.device)
        input_masks = self.value_dict["cond_frames_mask"].to(device=self.device)
        pluckers = self.value_dict["plucker_coordinate"].to(device=self.device)

        T = self.version_dict["T"]

        encoding_t = 1 
        with torch.no_grad():
            self.conditioner = self.conditioner.to(device=self.device)

            input_latents = self.ae.encode(imgs[input_masks], encoding_t)
            latents = torch.nn.functional.pad(
                input_latents, (0, 0, 0, 0, 0, 1), value=1.0
            )
            c_crossattn = repeat(self.conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)
            uc_crossattn = torch.zeros_like(c_crossattn)
            c_replace = latents.new_zeros(T, *latents.shape[1:])
            c_replace[input_masks] = latents
            uc_replace = torch.zeros_like(c_replace)
            c_concat= torch.cat(
                [
                    repeat(
                        input_masks,
                        "n -> n 1 h w",
                        h=pluckers.shape[2],
                        w=pluckers.shape[3],
                    ),
                    pluckers,
                ],
                1,
            )
            uc_concat = torch.cat(
                [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
            )
            c_dense_vector = pluckers
            uc_dense_vector = c_dense_vector
            c = {
                "crossattn": c_crossattn,
                "replace": c_replace,
                "concat": c_concat,
                "dense_vector": c_dense_vector,
            }
            uc = {
                "crossattn": uc_crossattn,
                "replace": uc_replace,
                "concat": uc_concat,
                "dense_vector": uc_dense_vector,
            }
            unload_model(self.conditioner)

            additional_model_inputs = {"num_frames": T}
            additional_sampler_inputs = {} # for no CFG, no additional inputs are needed

            return c, uc, additional_model_inputs, additional_sampler_inputs

    @torch.no_grad()
    def set_input_latent(self, ref_img):
        ref_latent = self.ae.encode(ref_img, 1)
        ref_latent = torch.nn.functional.pad(
                ref_latent, (0, 0, 0, 0, 0, 1), value=1.0
            )
        input_masks = self.value_dict["cond_frames_mask"].to(device=self.device)
        self.c["replace"][input_masks] = ref_latent
        
        # Set cross-attention conditioning
        self.conditioner = self.conditioner.to(device=self.device)
        c_crossattn = repeat(self.conditioner(ref_img).mean(0), "d -> n 1 d", n=self.version_dict["T"])
        self.c["crossattn"] = c_crossattn
        unload_model(self.conditioner)

    def prepare_input_for_denoiser(self, z_t, sigma):
        latent_shape = self.latent_shape
        if (z_t.shape[0] != latent_shape[0]): 
            z_t = torch.cat([torch.zeros((self.n_input_frames, latent_shape[1], latent_shape[2], latent_shape[3]), device="cuda"), z_t], dim=0) # filling shape to latent shape
            z_t = torch.cat([z_t, z_t[-1].unsqueeze(0).repeat(self.n_padding, 1, 1, 1)], dim=0)
        assert z_t.shape[0] == latent_shape[0]
        input, sigma, c = self.guider.prepare_inputs(z_t, sigma, self.c, self.uc)
        return input, sigma, c
    
    def get_sigma_from_timestep(self, timestep):
        return self.denoiser.idx_to_sigma(timestep.to(dtype=torch.int))

    def predict_eps_and_sample(self, z_t, timestep, guidance_scale=2.0):     
        sigma_orig = self.get_sigma_from_timestep(timestep)
        if sigma_orig.shape[0] == 1:
            s_in = torch.ones([self.latent_shape[0]], device="cuda")
            sigma_orig = sigma_orig.item() * s_in

        input, sigma, c = self.prepare_input_for_denoiser(z_t, sigma_orig)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_z0_c_and_uc = self.denoiser(self.model, input, sigma, c.copy(), **self.additional_model_inputs)

            # logic taken from seva's DiscreteDenoiser __call__() method, 
            # here we isolate the network output (which is the predicted noise) from the denoiser's return value.
            sigma = append_dims(sigma, input.ndim)
            if "replace" in c: # for input frames (mask = True) use the clean latents
                x, mask = c.pop("replace").split((input.shape[1], 1), dim=1)
                input = input * (1 - mask) + x * mask
            c_skip, c_out, _, _ = self.denoiser.scaling(sigma)
            pred_eps_c_and_uc = (pred_z0_c_and_uc - input * c_skip) / c_out

            pred_z0 = self.guider(pred_z0_c_and_uc, sigma_orig, guidance_scale, **self.additional_sampler_inputs)
            pred_eps = self.guider(pred_eps_c_and_uc, sigma_orig, guidance_scale, **self.additional_sampler_inputs)

        return pred_eps, pred_z0
    
    def get_only_output_frames(self, preds):
        assert preds.shape[0] == self.latent_shape[0]
        test_frames = preds[self.n_input_frames:len(preds)-self.n_padding]
        return test_frames

    def noise_to_timestep(self, z0, timestep, eps):
        eps  = eps / eps.var()
        sigma_zero = self.get_sigma_from_timestep(timestep*0.0)
        sigma_t = self.get_sigma_from_timestep(timestep)


        z_t = z0 + eps * append_dims(sigma_t**2 - sigma_zero**2, z0.ndim) ** 0.5

        return z_t

    def get_text_embeddings(self, s): #filler
        return None
    
    def euler_edm_step(self, latent, curr_t, next_t):
        # values used by seva/demo.py
        s_churn = 0.0
        s_noise = 1.0
        gamma = 0.0

        curr_sigma = self.get_sigma_from_timestep(curr_t)
        next_sigma = self.get_sigma_from_timestep(next_t)

        sigma_hat = curr_sigma * (gamma + 1.0) + 1e-6
        eps = torch.randn_like(latent) * s_noise
        latent = latent + eps * append_dims(sigma_hat**2 - curr_sigma**2, latent.ndim) ** 0.5

        _, z0_student = self.predict_eps_and_sample(latent, curr_t)
        d = to_d(latent, curr_sigma, z0_student)
        dt = append_dims(next_sigma - curr_sigma, latent.ndim)
        latent = (latent + dt * d)

        return latent, z0_student


    def decode(self, latent, type="np"):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = self.ae.decode(latent, 1).to(device="cpu")
        if type != "pt": # np or PIL
            raise NotImplementedError("Only 'pt' type is implemented currently.")
            #samples = seva_tensor_to_np_plottable(samples.detach())  
        if type == "PIL":
            samples = [Image.fromarray(sample) for sample in samples]

        return samples
    
    
class SevaStudent(SevaPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def set_model_requires_grad(self):
        self.model.requires_grad_(True)

    def get_trainable_parameters(self):
        return self.model.parameters()