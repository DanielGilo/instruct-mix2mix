import torch


# from DreamFusion appendix A1: sds_loss(x) = weight(t) * dot(stopgrad[epshat_t - eps], x) where x = g(theta)
# and its grad is the known formula: weight(t) * (epshat_t - eps) * grad(x)
def sds_loss(z0_student, teacher, teacher_guidance_scale, eps, timestep, w_t):
    with torch.inference_mode():
        z_t = teacher.noise_to_timestep(z0_student, timestep, eps)
        e_t, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep, teacher_guidance_scale)
        grad_z = w_t * (e_t - eps)
        grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
    sds_loss = grad_z.clone() * z0_student
    sds_loss = sds_loss / z0_student.numel() # Daniel: normalization is necessary to avoid exploding grads
    del grad_z
    sds_loss = sds_loss.sum() 

    return sds_loss, z_t, pred_z0