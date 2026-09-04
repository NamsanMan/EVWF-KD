# EVWF-KD

Reference implementation of EVWF-KD. Only the proposed method is included.

## Requirements

Python >= 3.10, PyTorch 2.8.0 (CUDA 12.9), transformers 4.57.1.

```
pip install -r requirements.txt
```

## Data

The degraded low-resolution images and the generated pseudo-depth maps are
available here:

> **Download:** https://drive.google.com/drive/folders/1Hb0rMXLWK0tbQU09UfaeD9Piz67i5pK9?usp=sharing


```
datasets/
  CamVid_LR/{A_set,B_set}/{train,val,test}/
      images/     provided
      labels/     supply from the original dataset
  CamVid_HR/{A_set,B_set}/{train,val,test}/
      images/     supply from the original dataset (training only)
      depths/     provided
  KITTI_LR, KITTI_HR, MiniCity_LR, MiniCity_HR: same structure
weights/
  camvid_A.pth    camvid_B.pth
  kitti_A.pth     kitti_B.pth
  minicity_A.pth  minicity_B.pth
```

The provided images define the split, so a file placed under `A_set/train`
must take the name of the provided image in the same directory.

Paths are read from environment variables; no file needs to be edited.

| variable | meaning | default |
| --- | --- | --- |
| `SWEEP_DATASET` | `camvid`, `kitti` or `minicity` | `kitti` |
| `SWEEP_FOLD` | `A_set` or `B_set` | `A_set` |
| `SWEEP_DATA_ROOT` | dataset root | `./datasets` |
| `SWEEP_RESULT_BASE_DIR` | output root | `./results` |

## Files

```
config.py               all settings
data_loader.py          dataset and augmentation
models/                 network wrappers
kd_engines/             distillation objectives
main_teacher.py         teacher entry point
main_kd.py              student entry point
train_teacher.py        teacher training loop
train_kd.py             student training loop
evaluate.py             evaluation metrics
eval_kd_checkpoint.py   evaluate a saved checkpoint
```

## Usage

Evaluate a checkpoint:

```
SWEEP_DATASET=camvid SWEEP_FOLD=A_set \
python eval_kd_checkpoint.py \
    --ckpt ./weights/camvid_A.pth \
    --model-name segformerb0 \
    --split test
```

Train the teacher, then the student:

```
SWEEP_DATASET=camvid SWEEP_FOLD=A_set \
SWEEP_PROJECT_NAME=camvid_A_teacher \
SWEEP_ENGINE_NAME=fdcs_gbst_teacher \
SWEEP_TEACHER_NAME=segformerb3_fdcs \
SWEEP_STUDENT_NAME=segformerb3_fdcs \
python main_teacher.py

SWEEP_DATASET=camvid SWEEP_FOLD=A_set \
SWEEP_PROJECT_NAME=camvid_A_student \
SWEEP_ENGINE_NAME=priv_reach_geo_dec_stagek_sgsc \
SWEEP_TEACHER_NAME=segformerb3_fdcs \
SWEEP_STUDENT_NAME=segformerb0 \
SWEEP_FREEZE_TEACHER=1 \
SWEEP_TEACHER_CKPT=./results/camvid_A_teacher/best_model.pth \
SWEEP_TRAIN_EPOCHS=300 \
python main_kd.py
```

Defaults are set in `config.py`.

## License

MIT. See [LICENSE](LICENSE).
