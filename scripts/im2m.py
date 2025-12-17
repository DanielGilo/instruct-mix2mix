import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
from typing import List
import os
from os.path import basename
import math
import torch
import torch.nn.functional as F

import gc
import numpy as np
import random
from scipy.stats import truncnorm
from torch.optim import AdamW
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from src.teachers import get_teacher
from src.students import SevaStudent
from src.loss import sds_loss
from src.seva_utils import get_value_dict_of_scene
from src.attention import CrossFrameAttnProcessor


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # avoids memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # necessary when setting torch.use_dererministic_algorithms(True), increases memory by 24MB

rng = np.random.default_rng(42)


def seed_everything(seed: int = 0):
    global rng
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed set to {seed}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse user arguments for an experiment.")

    parser.add_argument("--teacher_name",
                        type=str,
                        required=True,
                        help="Name of the Teacher.")
    parser.add_argument("--scene_path",
                        type=str,
                        required=True,
                        help="Path to the scene folder, containing transforms.json file")
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="Path to the experiment output directory where results will be saved.")
    parser.add_argument("--exp_name",
                        type=str,
                        default=None,
                        required=False,
                        help="Name of experiment (for logging purposes).")
    parser.add_argument( "--frame_indices",
                        type=lambda s: [int(item) for item in s.split(',')],
                        required=True,
                        help="Comma-separated list of frame indices (e.g., '1,11,43,35') corresponding to indices of frames in the transforms.json file.")
    parser.add_argument("--prompt",
                        type=str,
                        default="",
                        help="Original prompt describing the scene. Required for ControlNet teachers.")
    parser.add_argument("--editing_prompt",
                        type=str,
                        default="",
                        help="Instructional editing prompt. Required for pix2pix teacher.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Max learning rate.")
    parser.add_argument("--final_lr",
                        type=float,
                        default=5e-5,
                        help="Final learning rate after cosine decay.")
    parser.add_argument("--n_warmup_steps",
                        type=int,
                        default=200,
                        help="Number of linear LR warmup steps.")
    parser.add_argument("--n_distill_per_timestep",
                        type=int,
                        default=50,
                        help="Number of distillation iterations per timestep.")
    parser.add_argument("--n_distill_initial_timestep",
                        type=int,
                        default=200,
                        required=True,
                        help="Number of distillation iterations of the first timestep.")
    parser.add_argument("--distill_dt",
                        type=int,
                        default=25,
                        help="Timestep gap for the distillation process.")
    parser.add_argument("--t_min",
                        type=int,
                        default=25,
                        help="Minimal timestep for distillation.")
    parser.add_argument("--teacher_cfg",
                        type=float,
                        default=7.5,
                        help="Teacher CFG weight.")
    parser.add_argument("--image_cfg",
                        type=float,
                        default=1.5,
                        help="Pix2pix Teacher image CFG weight.")
    parser.add_argument("--do_compile",
                        action='store_true',
                        help="Boolean flag to compile models.")
    parser.add_argument("--use_grad_checkpointing",
                        action='store_true',
                        help="Enable gradient checkpointing for the SEVA model. Reduces VRAM usage at the cost of slower training/backward pass.")
    parser.add_argument(
                        "--teacher_timestep_shape_factor",
                        type=float,
                        default=0.5,
                        help=(
                            "The shape factor for the truncated normal distribution that the teacher timestep is sampled from.\n"
                            "A value of 0.5 would result in a roughly uniform distribution between student_t and T, and higher values"
                            " would result in a lower variance and higher probability around the lower bound -- student_t."
                        ))
    parser.add_argument("--log_every",
                        type=int,
                        default=10,
                        help="Steps between scalar logs/prints.")
    parser.add_argument("--image_log_every",
                        type=int,
                        default=200,
                        help="Steps between image logs.")
    parser.add_argument("--verbose",
                        action='store_true',
                        help="Verbose mode for detailed logging.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args


@dataclass
class TrainConfig:
    # core experiment info
    teacher_name: str
    scene_path: str
    output_path: str
    frame_indices: List[int]
    num_views: int
    exp_name: str

    # text prompts
    prompt: str
    editing_prompt: str

    # optimization
    lr: float
    final_lr: float
    n_warmup_steps: int
    n_distill_per_timestep: int
    n_distill_initial_timestep: int
    distill_dt: int
    t_min: int
    t_max: int

    # guidance
    teacher_cfg: float
    image_cfg: float

    # teacher sampling
    teacher_timestep_shape_factor: float

    # misc
    seed: int
    do_compile: bool
    device: torch.device
    use_grad_checkpointing: bool

    # logging
    log_every: int
    image_log_every: int
    verbose: bool


def build_config(args) -> TrainConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return TrainConfig(
        teacher_name=args.teacher_name,
        scene_path=args.scene_path,
        output_path=args.output_path,
        frame_indices=args.frame_indices,
        num_views=len(args.frame_indices),
        exp_name=args.exp_name,
        prompt=args.prompt,
        editing_prompt=args.editing_prompt,
        lr=args.lr,
        final_lr=args.final_lr,
        n_warmup_steps=args.n_warmup_steps,
        n_distill_per_timestep=args.n_distill_per_timestep,
        n_distill_initial_timestep=args.n_distill_initial_timestep,
        distill_dt=args.distill_dt,
        t_max=999,
        t_min=args.t_min,
        teacher_cfg=args.teacher_cfg,
        image_cfg=args.image_cfg,
        teacher_timestep_shape_factor=args.teacher_timestep_shape_factor,
        seed=args.seed,
        do_compile=args.do_compile,
        device=device,
        log_every=args.log_every,
        image_log_every=args.image_log_every,
        verbose=args.verbose,
        use_grad_checkpointing=args.use_grad_checkpointing,
    )


# ------------- TensorBoard helpers ------------- #

def setup_writer(config: TrainConfig) -> SummaryWriter:
    if config.exp_name is None:
        exp_name = f"{config.teacher_name}-{basename(config.scene_path)}-nv-{config.num_views}"
    else:
        exp_name = config.exp_name
        
    log_dir = os.path.join("runs", exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    return writer


def log_initial_images(writer: SummaryWriter, value_dict, ref_img: torch.Tensor):
    """
    Log ground-truth sequence and reference (teacher) image at the beginning.
    """
    all_gt = value_dict["cond_frames"]  # [-1, 1], shape [N, C, H, W]
    gt_vis = (all_gt * 0.5 + 0.5).clamp(0, 1).cpu()
    gt_vis_no_ref = gt_vis[1:]  # remove input view for logging
    grid = vutils.make_grid(gt_vis_no_ref, nrow=min(4, gt_vis.shape[0]))
    writer.add_image("gt/sequence", grid, 0)

    # ref_img is assumed to be [-1, 1], shape [1, C, H, W] or [C, H, W]
    if ref_img.dim() == 4:
        ref = ref_img[0]
    else:
        ref = ref_img
    ref_vis = (ref * 0.5 + 0.5).clamp(0, 1).cpu()
    writer.add_image("teacher/reference_image", ref_vis, 0)

    print("[TB] Logged initial GT sequence and reference image.")


def log_step(
    writer: SummaryWriter,
    config: TrainConfig,
    iter_count: int,
    time_i: int,
    n_timesteps: int,
    distill_i: int,
    n_distill_iters: int,
    loss_value: float,
    optimizer,
    student_t: torch.Tensor,
):
    """
    Print progress and log scalars to TensorBoard.
    """
    lr = optimizer.param_groups[0]["lr"]
    t_int = int(student_t[0].item())

    # Console print
    if config.verbose:
        print(
            f"distill iteration: {distill_i + 1}/{n_distill_iters}, "
            f"timestep: {time_i + 1}/{n_timesteps}, "
            f"total iter: {iter_count}, "
            f"loss: {loss_value:.6e}, lr: {lr:.6e}, t: {t_int}"
        )

    # TensorBoard scalars
    if writer is not None:
        writer.add_scalar("train/loss", loss_value, iter_count)
        writer.add_scalar("train/lr", lr, iter_count)
        writer.add_scalar("train/timestep", t_int, iter_count)


def log_student_predictions(
    writer: SummaryWriter,
    config: TrainConfig,
    iter_count: int,
    teacher,
    student_pred: torch.Tensor,
    teacher_pred_z0: torch.Tensor,
):
    """
    Occasionally log decoded student and teacher predictions as images.

    - student_pred: latent in teacher space (B, C, H, W)
    - teacher_pred_z0: latent in teacher space (B, C, H, W)
    """
    if writer is None:
        return

    image_log_every = getattr(config, "image_log_every", 500)

    if (iter_count > 1) and (iter_count % image_log_every != 0):
        return

    with torch.no_grad():
        # Decode both with teacher's decoder
        student_latents = student_pred.to(device=config.device, dtype=torch.float16)
        teacher_latents = teacher_pred_z0.to(device=config.device, dtype=torch.float16)

        student_imgs_np = teacher.decode(student_latents, type="np")  # [B, H, W, C] or similar
        teacher_imgs_np = teacher.decode(teacher_latents, type="np")

        student_imgs = torch.from_numpy(student_imgs_np)
        teacher_imgs = torch.from_numpy(teacher_imgs_np)

        # If images are [B, H, W, C], convert to [B, C, H, W]
        if student_imgs.ndim == 4 and student_imgs.shape[-1] in (1, 3, 4):
            student_imgs = student_imgs.permute(0, 3, 1, 2)
        if teacher_imgs.ndim == 4 and teacher_imgs.shape[-1] in (1, 3, 4):
            teacher_imgs = teacher_imgs.permute(0, 3, 1, 2)

        # Clamp to [0, 1] if decoder returns that range
        student_imgs = student_imgs.clamp(0.0, 1.0)
        teacher_imgs = teacher_imgs.clamp(0.0, 1.0)

        max_imgs = min(4, student_imgs.shape[0], teacher_imgs.shape[0])

        student_grid = vutils.make_grid(student_imgs[:max_imgs].cpu(), nrow=max_imgs)
        teacher_grid = vutils.make_grid(teacher_imgs[:max_imgs].cpu(), nrow=max_imgs)

        writer.add_image("predictions/student", student_grid, iter_count)
        writer.add_image("predictions/teacher", teacher_grid, iter_count)

        if config.verbose:
            print(f"[TB] Logged student & teacher predictions at iter {iter_count}.")


# ------------- Model / training helpers ------------- #

def build_models(config, value_dict):
    all_gt = value_dict["cond_frames"]  # [-1, 1]
    num_inputs = 1 
    torch_gt = (((all_gt[num_inputs:value_dict["num_imgs_no_padding"]] + 1) / 2.0) * 255).clamp(0, 255).to(torch.uint8)
    pil_gt = [to_pil_image(torch_gt[i]) for i in range(torch_gt.shape[0])]

    student = SevaStudent(
        device=config.device,
        value_dict=value_dict,
        do_compile=config.do_compile,
        use_grad_checkpointing=config.use_grad_checkpointing,
    )

    teacher = get_teacher(
        config.teacher_name,
        prompt=config.prompt,
        editing_prompt=config.editing_prompt,
        image_guidance_scale=config.image_cfg,
        device=config.device,
        gt_images=pil_gt,
        do_compile=config.do_compile,
    )

    return student, teacher


@torch.no_grad()
def prepare_reference_image(config, teacher, value_dict):
    all_gt = value_dict["cond_frames"]  # [-1, 1]
    teacher_res = (teacher.pixel_space_shape[1], teacher.pixel_space_shape[2])  # (H, W)
    student_res = value_dict["version_dict"]["H"], value_dict["version_dict"]["W"]

    input_img = ((all_gt[0] + 1) / 2.0).clamp(0, 1)  # scale to [0, 1]
    input_img = F.interpolate(input_img.unsqueeze(0), size=teacher_res, mode="bilinear", align_corners=False)
    input_img = input_img.to(torch.float16)  # ensure float16

    pipe = teacher.pipeline
    pipe = pipe.to(config.device)

    if config.teacher_name == "pix2pix":
        ref_img_pil = pipe(
            prompt=config.editing_prompt,
            image=to_pil_image(input_img[0].cpu()),
            guidance_scale=config.teacher_cfg,
            image_guidance_scale=config.image_cfg,
        )[0]
    else:
        ref_img_pil = pipe(
            prompt=config.prompt,
            image=to_pil_image(input_img[0].cpu()),
            guidance_scale=config.teacher_cfg,
        )[0]

    ref_img = F.interpolate(
        pil_to_tensor(ref_img_pil[0]).unsqueeze(0),
        size=student_res,
        mode="bilinear",
        align_corners=False,
    ) / 255.0
    ref_img = ref_img * 2 - 1.0  # scale to [-1, 1]
    ref_img = ref_img.to(config.device)

    return ref_img


def build_timesteps(config):
    """
    Build the 1D tensor of timesteps used for distillation.
    """
    t_max = config.t_max
    t_min = config.t_min

    n_distill_timesteps = ((t_max + 1 - t_min) // config.distill_dt) + 1

    timesteps = torch.linspace(
        t_max,
        t_min,
        steps=n_distill_timesteps,
        device=config.device,
        dtype=torch.int64,
    )

    return timesteps


def init_latent(config, student):
    """
    Initialize the latent and the s_in vector used in the distillation loop.
    """
    device = config.device

    latent_shape = student.latent_shape
    latent = torch.randn(latent_shape, device=device)

    disc = student.discretization(1000, device=device)
    latent *= torch.sqrt(1.0 + disc[0] ** 2.0)

    s_in = latent.new_ones([latent.shape[0]])

    return latent, s_in


def build_optimizers(config, student):
    """
    Build optimizer, LR scheduler and GradScaler.
    """
    params = list(student.get_trainable_parameters())
    optimizer = AdamW(params=params, lr=config.lr, weight_decay=0.01)

    max_lr = config.lr
    final_lr = config.final_lr
    warmup_steps = config.n_warmup_steps
    n_distill_per_timestep = config.n_distill_per_timestep
    distill_dt = config.distill_dt
    min_lr = getattr(config, "min_lr", 1e-9)

    # decay until t=0 even if t_min > 0
    decay_steps = int((1000 / distill_dt) * n_distill_per_timestep - warmup_steps)

    def lr_lambda(step: int) -> float:
        # Linear warmup from min_lr -> max_lr
        if step < warmup_steps and warmup_steps > 1:
            alpha = step / (warmup_steps - 1)
            lr_abs = (1.0 - alpha) * min_lr + alpha * max_lr
            return lr_abs / max_lr

        if warmup_steps <= 1 and step < warmup_steps:
            return 1.0

        # Cosine decay from max_lr -> final_lr
        if step < warmup_steps + decay_steps and decay_steps > 0:
            decay_step = step - warmup_steps
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
            lr_abs = final_lr + (max_lr - final_lr) * cosine_decay
            return lr_abs / max_lr

        # Constant final_lr afterwards
        return final_lr / max_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    return optimizer, scheduler, scaler


def get_n_distill_iters_for_t(t, config):
    t_int = int(t)
    if t_int == 999:
        return config.n_distill_initial_timestep
    if t_int >= config.t_min:
        return config.n_distill_per_timestep
    return 0


def upper_skewed_truncated_normal(lower_bound, upper_bound, shape_factor=3.0, size=1000):
    """
    Samples from a truncated Gaussian shifted towards *upper_bound*.
    """
    if lower_bound >= upper_bound:
        return np.array([upper_bound])

    mean = upper_bound
    scale = (mean - lower_bound) / shape_factor

    a = (lower_bound - mean) / scale
    b = 0  # because mean == upper_bound

    return truncnorm.rvs(a=a, b=b, loc=mean, scale=scale, size=size, random_state=rng).astype(int)


def distill_step(
    config,
    student,
    teacher,
    latent,
    student_t,
    optimizer,
    scheduler,
    scaler,
):
    """
    Perform a single distillation optimization step for a given student timestep.

    Returns:
        loss_value (float), student_pred, teacher_pred_z0
    """
    device = config.device

    aggressive_gc = getattr(config, "aggressive_gc", False)
    if aggressive_gc:
        gc.collect()
        torch.cuda.empty_cache()

    optimizer.zero_grad(set_to_none=True)

    # 1. Student forward: predict z0
    _, z0_student = student.predict_eps_and_sample(latent, student_t, config.teacher_cfg)
    if not torch.isfinite(z0_student).all():
        raise FloatingPointError("Non-finite values in student prediction (z0_student).")

    z0_student_out_frames = student.get_only_output_frames(z0_student)

    # Scalar timestep tensor for teacher
    timestep = torch.as_tensor(
        student_t[0].item(),
        dtype=torch.int64,
        device=device,
    ).view(1)

    # Weight for loss
    w_t = 1.0

    # 2. Sample teacher timestep (truncated normal)
    teacher_upper_bound = getattr(config, "teacher_timestep_upper_bound", 950)

    teacher_timestep_np = upper_skewed_truncated_normal(
        lower_bound=int(timestep.item()),
        upper_bound=teacher_upper_bound,
        shape_factor=config.teacher_timestep_shape_factor,
        size=1,
    )[0]

    teacher_timestep = torch.full_like(
        timestep,
        int(teacher_timestep_np),
        dtype=torch.int64,
        device=device,
    )

    # 3. Set random cross-frame attention key frame index
    CrossFrameAttnProcessor.key_frame_index = torch.randint(
        0, z0_student_out_frames.shape[0], (1,), device=device
    ).item()

    # 4. Resize student prediction to teacher space
    target_h, target_w = teacher.latent_shape[1:]
    student_pred = F.interpolate(
        z0_student_out_frames,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).to(dtype=teacher.dtype, device=device)

    eps = torch.randn_like(student_pred, device=device)

    # 5. Compute loss
    loss, z_t, teacher_pred_z0 = sds_loss(
        student_pred,
        teacher,
        config.teacher_cfg,
        eps,
        teacher_timestep,
        w_t,
    )

    if aggressive_gc:
        gc.collect()
        torch.cuda.empty_cache()

    # 6. Backprop + optimizer step + scheduler
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]

    max_norm = getattr(config, "grad_clip_norm", 0.5)
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    scaler.step(optimizer)
    scaler.update()

    scheduler.step()

    loss_value = float(loss.item())
    return loss_value, student_pred.detach(), teacher_pred_z0.detach()


def euler_step_if_needed(
    config,
    student,
    latent,
    student_t,
    timesteps,
    time_index: int,
    s_in,
):
    """
    Perform an Euler-EDM step to the next timestep, unless we're at the last one.

    Returns:
        latent: updated latent (or unchanged if this was the last timestep)
    """
    n_timesteps = len(timesteps)

    if time_index >= n_timesteps - 1:
        return latent

    next_t = timesteps[time_index + 1] * s_in

    with torch.no_grad():
        latent, _ = student.euler_edm_step(
            latent,
            student_t,
            next_t,
        )

    if getattr(config, "verbose", True):
        print(f"t = {int(next_t[0])}")

    return latent


def train(config, student, teacher, value_dict, writer: SummaryWriter):
    timesteps = build_timesteps(config)
    latent, s_in = init_latent(config, student)
    optimizer, scheduler, scaler = build_optimizers(config, student)

    teacher.set_attn_processor("cross_frame")

    # log initial images (GT sequence + teacher reference)
    ref_img = student.input_latent if hasattr(student, "input_latent") else None
    if ref_img is not None:
        log_initial_images(writer, value_dict, ref_img)

    iter_count = 0
    last_student_pred = None  # for logging final result

    for time_i, t_scalar in enumerate(timesteps):
        student_t = t_scalar * s_in
        n_distill_iters = get_n_distill_iters_for_t(t_scalar, config)

        for distill_i in range(n_distill_iters):
            loss_value, student_pred, teacher_pred_z0 = distill_step(
                config=config,
                student=student,
                teacher=teacher,
                latent=latent,
                student_t=student_t,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            iter_count += 1
            last_student_pred = student_pred  # keep last student prediction

            # scalar logging + prints every config.log_every
            if iter_count % config.log_every == 0:
                log_step(
                    writer=writer,
                    config=config,
                    iter_count=iter_count,
                    time_i=time_i,
                    n_timesteps=len(timesteps),
                    distill_i=distill_i,
                    n_distill_iters=n_distill_iters,
                    loss_value=loss_value,
                    optimizer=optimizer,
                    student_t=student_t,
                )

            # occasional image logging (student + teacher predictions)
            log_student_predictions(
                writer=writer,
                config=config,
                iter_count=iter_count,
                teacher=teacher,
                student_pred=student_pred,
                teacher_pred_z0=teacher_pred_z0,
            )

        latent = euler_step_if_needed(
            config=config,
            student=student,
            latent=latent,
            student_t=student_t,
            timesteps=timesteps,
            time_index=time_i,
            s_in=s_in,
        )

   # ---- Final student prediction logging and saving ----
    if last_student_pred is not None:
        with torch.no_grad():
            latents = last_student_pred.to(device=config.device, dtype=torch.float16)
            final_imgs_np = teacher.decode(latents, type="np")
            final_imgs = torch.from_numpy(final_imgs_np)

            # If images are [B, H, W, C], convert to [B, C, H, W]
            if final_imgs.ndim == 4 and final_imgs.shape[-1] in (1, 3, 4):
                final_imgs = final_imgs.permute(0, 3, 1, 2)

            final_imgs = final_imgs.clamp(0.0, 1.0)
            
            output_dir = Path(config.output_path)
            edited_frames_dir = output_dir / "edited_frames"
            edited_frames_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Save individual edited frames
            for i, img_tensor in enumerate(final_imgs):
                frame_path = edited_frames_dir / f"frame_{i+1:04d}.png"
                vutils.save_image(img_tensor, str(frame_path))

            # 2. Save the final grid
            max_imgs = min(4, final_imgs.shape[0])
            final_grid = vutils.make_grid(final_imgs[:max_imgs].cpu(), nrow=max_imgs)
            grid_path = output_dir / "final_grid.png"
            vutils.save_image(final_grid, str(grid_path))

            # 3. Log to TensorBoard
            if writer is not None:
                writer.add_image("final/student", final_grid, iter_count)

            if config.verbose:
                print(f"[TB] Logged final student prediction at iter {iter_count}.")
                print(f"[Save] Saved {len(final_imgs)} edited frames and final grid to {output_dir}")


def main() -> None:
    args = parse_arguments()
    config = build_config(args)
    seed_everything(config.seed)

    value_dict = get_value_dict_of_scene(Path(config.scene_path), 
                                        frame_indices=config.frame_indices)
    student, teacher = build_models(config, value_dict)

    ref_img = prepare_reference_image(config, teacher, value_dict)
    student.set_input_latent(ref_img)
    # keep a reference for logging
    student.input_latent = ref_img

    writer = setup_writer(config)
    try:
        train(config, student, teacher, value_dict, writer)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
