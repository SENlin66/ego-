# SAM3 and stereo-depth experiment code (2026-09-28)

这是 02994 实验的**源码整理版**：包括 SAM3 分割/跟踪，以及
FoundationStereo、RAFT-Stereo 的推理适配与对比视频脚本。
不含原始数据、模型权重、第三方完整仓库或生成的视频。先看下方模型获取方式，
再按示例命令在有 GPU 的服务器运行；本机没有重新跑模型。

This folder is a source-only snapshot of the 02994 experiment. It contains our
inference adapters and data preparation, not the pretrained models, weights,
episode files, or generated videos. Run commands from this folder on a machine
with a suitable NVIDIA GPU. All output paths below stay inside this folder.

## Contents

| File | Purpose |
| --- | --- |
| `sam3/segment_seed.py` | Prompt SAM3 on one RGB frame and save its binary mask and overlay. |
| `sam3/propagate_masks.py` | Track the mask over RGB frames in overlapping chunks; save masks, overlays, video and summary. |
| `stereo/prepare_rgbd.py` + `stereo/stereo_geometry.py` | Decode MCAP, pair and rectify stereo frames, and produce the original SGBM RGB-D baseline. |
| `stereo/prepare_pairs.py` | Export the same rectified frames for both learned stereo models. |
| `stereo/run_model.py` | Load FoundationStereo or RAFT-Stereo and convert disparity to metric depth. |
| `stereo/Utils.py` | Minimal inference compatibility helpers used by FoundationStereo without its optional Open3D import. |
| `stereo/make_comparison.py` | Make the RGB / SGBM / FoundationStereo / RAFT-Stereo comparison video. |

The original project files were left untouched. The copies here accept
external input paths so this folder can be published separately.

## Inputs and dependencies

Use Python 3.12 and separate project-local environments for SAM3 and stereo.
Install the CUDA-enabled PyTorch build appropriate for your driver/GPU before
the requirements files. The 02994 run used a CUDA device visible as device 0
inside the process. On a shared server, select an available GPU explicitly
with `CUDA_VISIBLE_DEVICES`; these scripts do not change drivers, Docker, or
system Python. Check each upstream repository's own dependency instructions
as well.

```bash
python3 -m venv .venv-sam3
.venv-sam3/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-sam3/bin/python -m pip install -r requirements-sam3.txt

python3 -m venv .venv-stereo
.venv-stereo/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-stereo/bin/python -m pip install -r requirements-stereo.txt
```

The PyTorch commands above match the CUDA 12.8 stack used by this project;
verify driver compatibility and the upstream models' requirements before
using a different GPU. The current local checkout lacks the OpenCV and MCAP
packages, so this source export has passed syntax and CPU-only helper checks,
but its GPU inference has not been rerun here.

Obtain these separately and place them at:

- SAM3 model snapshot: `sam3_model/` (or pass any directory via
  `--model`). It must include `model.safetensors` and the Transformers
  configuration/processor files. Source:
  [facebook/sam3](https://huggingface.co/facebook/sam3), revision
  `3c879f39826c281e95690f02c7821c4de09afae7`; its access and license
  terms apply.
- FoundationStereo source: `stereo/models/FoundationStereo/`. Source:
  [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo),
  commit `6e8806816b533e4d13ddbb95ffa907b797060a62`.
  Include its local `dinov2/` dependency. Its NVIDIA license applies.
- FoundationStereo checkpoint and config:
  `stereo/models/checkpoints/23-51-11/model_best_bp2.pth` and
  `stereo/models/checkpoints/23-51-11/cfg.yaml`.
- RAFT-Stereo source: `stereo/models/RAFT-Stereo/`. Source:
  [princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo),
  commit `6e93ed2169bd858dbb43033988563f3b0bb49506`.
  Its MIT license applies.
- RAFT-Stereo checkpoint:
  `stereo/models/checkpoints/raftstereo-middlebury.pth`.

Checkpoint names are those used in this experiment. Obtain weights through
the official project instructions; this folder has no weight downloader.
`stereo/deps/` can hold project-local Python packages when the server
environment cannot install them into the virtual environment.

## Reproduce the 02994 comparison

Set `MCAP` to the 02994 MCAP path and `META` to the 2942
`device_meta.json` path. The original 02994 run used that 2942 calibration:
its stereo geometry passed consistency checks, but its binding to 02994 was
not independently proven. Do not treat the resulting depth as ground truth.

```bash
export MCAP=/path/to/episode_02994_2026_08_06_12_12_57.mcap
export META=/path/to/2942/device_meta.json
export CUDA_VISIBLE_DEVICES=1  # example only: first check which GPU is free

.venv-stereo/bin/python stereo/prepare_rgbd.py \
  --mcap "$MCAP" --meta "$META" --start-relative-s 20 --duration-s 10 \
  --output stereo/rgbd

.venv-stereo/bin/python stereo/prepare_pairs.py \
  --mcap "$MCAP" --meta "$META" --rgbd stereo/rgbd \
  --start-s 20 --end-s 30 --expected-frames 299

.venv-stereo/bin/python stereo/run_model.py --model foundation --iters 32
.venv-stereo/bin/python stereo/run_model.py --model raft --iters 32
.venv-stereo/bin/python stereo/make_comparison.py
.venv-stereo/bin/python stereo/make_comparison.py --zoom
```

The example above only prepares the 20-30 s window. The
`prepare_pairs.py` selection still uses original MCAP-relative times.
To run SAM3 over the entire episode, prepare a separate full RGB sequence:
`--start-relative-s 0 --output stereo/rgbd_full` (omit
`--duration-s`). This decoder holds the decoded window in memory and a
full episode can require tens of GB of RAM. Then point SAM3
`--image` and `--dataset` at `stereo/rgbd_full`.

`prepare_rgbd.py` writes rectified cam0 RGB and SGBM depth in 16-bit
millimeters. `prepare_pairs.py` checks every rectified cam0 frame against
that baseline before exporting the matching cam1 frame. The model adapter
saves a disparity `.npy` and a uint16 depth PNG for each pair. A zero depth
means invalid, not a measured distance. The valid range is 0.15-3.0 m;
the comparison video clamps displayed colors beyond 1.0 m and paints invalid
pixels black. Valid-pixel coverage and visual smoothness are not accuracy
measurements.

The comparison videos appear in `stereo/output/`. The script expects both
learned models and SGBM depth for every row in `stereo/input/pairs.csv`.
To run a different window, change `--start-s` and `--end-s`, and omit
`--expected-frames`; the display crop remains tuned to 1024x768 02994 frames.

## Run SAM3 on the RGB frames

The commands below run on the prepared 20-30 s window. For the full episode,
first create `stereo/rgbd_full` as described above and replace
`stereo/rgbd` with `stereo/rgbd_full` in both commands.

```bash
.venv-sam3/bin/python sam3/segment_seed.py \
  --image stereo/rgbd/rgb/000000.png --model sam3_model \
  --output sam3_seed --box 820 430 950 565

.venv-sam3/bin/python sam3/propagate_masks.py \
  --dataset stereo/rgbd --model sam3_model \
  --seed-mask sam3_seed/right_gripper_mask.png \
  --output sam3_full --chunk-frames 60
```

Inspect that RGB image and choose a box around the target; the numbers above
are only an example prompt and are not a verified 02994 box. The 02994 full
run used 60-frame chunks,
one-frame overlap, FP16 on CUDA, and no keyframe resegmentation. Its recorded
summary reported 2253/2253 frames; this copy has not been rerun. The outputs
are `sam3_full/masks/`, `sam3_full/overlays/`,
`sam3_full/sam3_housing_mask_30hz.mp4`, and `sam3_full/summary.json`.
These are segmentation masks, not 6DoF pose estimates.

## Scope and publication

This folder does not run FoundationPose and does not produce a canonical
`6dof.csv`. In the larger pose pipeline, SAM3 masks and stereo depth are
observations only. Formal pose output uses
`T_cam0_from_right_gripper_housing` with XYZW quaternions; failed quality
gates leave frames empty.

The upstream model source and weights are deliberately excluded. Review
their licenses before distributing any upstream material. The local
`.gitignore` excludes raw MCAP, checkpoints, videos, generated inputs and
outputs; check `git status` before publishing.

CPU-only helper test: `python -m unittest discover -s tests -v`.
