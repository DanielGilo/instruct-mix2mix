# InstructMix2Mix

**Official implementation of "InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization."**

### [Project Page](https://danielgilo.github.io/instruct-mix2mix/) | [Research Paper](https://arxiv.org/pdf/2511.14899)

**TL;DR:** I-Mix2Mix performs instruction-driven edits on a sparse set of views. The key idea is SDS with a twist: we distill a 2D editor into a pretrained multi-view diffusion model rather than a NeRF/3DGS. The student’s learned 3D prior enables multi-view consistent edits, despite the sparse input.

---

## 🛠 Installation

We recommend using **Conda** to manage your environment for stability and isolation.

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/DanielGilo/instruct-mix2mix.git
cd instruct-mix2mix

```

### 2. Environment Setup

```bash
conda create -n im2m python=3.10 -y
conda activate im2m

```

### 3. Install PyTorch

Install PyTorch 2.6.0. Modify the CUDA version index (e.g., `cu124`) if required for your hardware:

```bash
pip install torch==2.6.0 torchvision --index-url "https://download.pytorch.org/whl/cu124"

```

### 4. Install SEVA & Requirements

I-Mix2Mix is built upon the SEVA framework. Install the submodule in editable mode and the remaining packages:

```bash
pip install -e ./external/stable-virtual-camera/
pip install -r requirements.txt

```

### 🔑 Model Access

This project relies on SEVA model weights. You must obtain access as detailed in the [SEVA Usage Documentation](https://github.com/Stability-AI/stable-virtual-camera?tab=readme-ov-file#open_book-usage).

---

## 📥 Input Format

I-Mix2Mix expects a **NeRF-style** `transforms.json` directory containing camera intrinsics and a `frames` list specifying per-frame extrinsics and image file paths. A subset of these frames is selected for editing via user-provided indices.

* **Sample Data:** See the `./data/` directory for reference scenes.
* **Preprocessing:** All input images undergo the SEVA pipeline:
1. **Resizing:** Automatic scaling if the shorter side is < 576 pixels.
2. **Center Cropping:** Images are cropped to **576 x 576** to match the model's training distribution.



---

## 🚀 Usage



### Core Arguments

| Argument | Description |
| --- | --- |
| `--scene_path` | Path to the directory containing `transforms.json`. |
| `--frame_indices` | Comma-separated indices (e.g., `"0,1,2,3"`) mapping to `transforms.json`. |
| `--teacher_name` | `"pix2pix"` for editing; `"depth"` or `"canny"` for ControlNet-based tasks. |
| `--editing_prompt` | The edit instruction used by the `pix2pix` teacher. |
| `--teacher_cfg` | Textual guidance scale (Higher = stronger edits; default: 7.5). |
| `--image_cfg` | Image guidance scale for `pix2pix` (Higher = source fidelity; default: 1.5). |

### Example Execution

```bash
python scripts/im2m.py \
     --scene_path ./data/person-small/ \
     --output_path ./outputs/knight/ \
     --frame_indices "0,1,2,3" \
     --editing_prompt "turn him into a knight" \
     --exp_name "Knight" \
     --teacher_name "pix2pix" \
     --lr 1.0e-4 \
     --final_lr 5.0e-5 \
     --n_distill_per_timestep 50 \
     --distill_dt 25 \
     --t_min 25 \
     --teacher_cfg 6.0 \
     --image_cfg 1.5 \
     --verbose

```

---

## 📂 Output Format

Results are stored in `--output_path`. A temporary `train_test_split_1.json` is generated for the parser during runtime and is automatically cleaned up.

```text
output_directory/
├── edited_frames/
│   ├── frame_0001.png   
│   ├── frame_0002.png  
│   └── ...
└── final_grid.png       # Consolidated multi-view visualization

```

---

## 💡 Tips & FAQs

1. **Input Verification:** Check the `gt` sequence logged to TensorBoard at the start of distillation. These images represent your input views **after** SEVA preprocessing (resizing and center-cropping). Ensure the subject of interest is well-contained within these crops.
2. **Validating Edits:** We inherit the limitations of InstructPix2Pix. Before full distillation, verify the 2D edit quality by checking the reference image in TensorBoard under `teacher/reference_image`.
3. **Hyperparameters:**
* **Guidance:** We recommend `[6.0, 9.0]` for textual CFG and `[1.5, 1.8]` for image CFG.
* **Consolidation:** Scenes with more than 4 frames may require increasing `--n_distill_per_timestep` to ensure consistency.


4. **Memory:** Use `--use_grad_checkpointing` to reduce VRAM usage on smaller GPUs.

---

## ✍️ Citation

If you find this work useful for your research, please cite:

```bibtex
@misc{gilo2025instructmix2mixconsistentsparseviewediting,
      title={InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization}, 
      author={Daniel Gilo and Or Litany},
      year={2025},
      eprint={2511.14899},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.14899}, 
}

```