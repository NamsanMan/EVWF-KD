from pathlib import Path
import numpy as np
import os
import json

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None or v == "" else int(v)

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or v == "" else float(v)

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.lower() in ("1", "true", "yes")

def _env_json(name: str, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else json.loads(v)

# ──────────────────────────────────────────────────────────────────
# 0. DATASET: 데이터셋 프로파일 (CamVid / MiniCity / KITTI)
# ──────────────────────────────────────────────────────────────────
# SWEEP_DATASET으로 전환한다. CamVid 실험을 다시 돌릴 때는
# SWEEP_DATASET=camvid를 명시해야 예전 설정 그대로 재현된다.
#
# 프로파일이 바꾸는 것: 데이터 폴더명, 클래스 정의, 입력 해상도, 학습 크롭, 배치.
# 그 외(옵티마이저, KD 엔진, 손실)는 데이터셋과 무관하므로 공유한다.
DATASET = _env_str("SWEEP_DATASET", "kitti").lower()
FOLD = _env_str("SWEEP_FOLD", "A_set")

# HR 교사 이미지 + pseudo-depth(= privileged 입력)를 로더가 함께 실어보낼지.
# 배치가 ((x_lr, x_hr, depth), mask) 로 나온다.
#
# 공개된 두 entry point(main_teacher.py, main_kd.py)는 모두 True 를 요구한다.
# 두 engine 다 3-튜플 배치를 전제하므로 False 는 지원하지 않는다.
USE_PRIVILEGED_INPUTS = True

# Cityscapes 19 trainId 순서 + Void(19). MiniCity 라벨은 이미 trainId로 저장돼 있다.
_MINICITY_CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "TrafficLight", "TrafficSign", "Vegetation", "Terrain",
    "Sky", "Person", "Rider", "Car", "Truck",
    "Bus", "Train", "Motorcycle", "Bicycle", "Void",
]

# KITTI는 CamVid와 같은 12클래스 체계(0~10 + Void 11)를 쓰므로 이름과 팔레트를
# 그대로 재사용한다. multi_weather_kitti의 seg_label을 실제로 집계해 확인했다.
_CAMVID_CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Sidewalk",
    "Tree", "SignSymbol", "Fence", "Car",
    "Pedestrian", "Bicyclist", "Void",
]
_CAMVID_CLASS_COLORS = np.array([
    [128, 128, 128],  # Sky
    [128, 0, 0],      # Building
    [192, 192, 128],  # Pole
    [128, 64, 128],   # Road
    [0, 0, 192],      # Sidewalk
    [128, 128, 0],    # Tree
    [192, 128, 128],  # SignSymbol
    [64, 64, 128],    # Fence
    [64, 0, 128],     # Car
    [64, 64, 0],      # Pedestrian
    [0, 128, 192],    # Bicyclist
    [0, 0, 0],        # Void
], dtype=np.uint8)

_DATASET_PROFILES = {
    "camvid": {
        "hr_dirname": "CamVid_HR",
        "lr_dirname": "CamVid_LR",
        # 원본 720x960의 절반. LR 90x120을 4배 업샘플한 크기와 같다.
        "input_resolution": (360, 480),
        # None이면 기존 방식(80~100% 스케일 크롭 후 원래 크기로 복원)을 그대로 쓴다.
        "train_crop": None,
        "batch_size": 8,
        # 316 % 8 = 4, 315 % 8 = 3. 마지막 배치가 충분히 커서 버릴 이유가 없다.
        "drop_last": False,
        "ignore_index": 11,
        "class_names": _CAMVID_CLASS_NAMES,
        "class_colors": _CAMVID_CLASS_COLORS,
    },
    "kitti": {
        "hr_dirname": "KITTI_HR",
        "lr_dirname": "KITTI_LR",
        # 원본은 1241x376. LR 310x94를 4배 업샘플하면 1240x376이므로 가로를
        # 1픽셀만 줄여 맞춘다. 라벨은 1241 그대로 저장돼 있고 로더가 리사이즈한다.
        "input_resolution": (376, 1240),
        # 크롭하지 않는다. 높이가 376뿐이라 큰 정사각 크롭이 불가능하고,
        # 전체 프레임 x batch 4 = 1.86M px/step으로 MiniCity 크롭(1.64M)과 비슷해
        # 메모리도 감당된다. 무엇보다 CamVid와 같은 증강 경로를 쓰게 되어
        # MiniCity에서 겪은 train/test 컨텍스트 불일치가 생기지 않는다.
        "train_crop": None,
        "batch_size": 4,
        # A_set train 201 % 4 = 1. 마지막 배치가 1장이면 BatchNorm 통계가 무의미해진다.
        "drop_last": True,
        "ignore_index": 11,
        # KITTI 라벨도 0~11로 CamVid와 같은 체계다.
        "class_names": _CAMVID_CLASS_NAMES,
        "class_colors": _CAMVID_CLASS_COLORS,
    },
    "minicity": {
        "hr_dirname": "MiniCity_HR",
        "lr_dirname": "MiniCity_LR",
        # 원본(=라벨) 해상도. LR 256x512를 4배 업샘플한 크기와 같다.
        "input_resolution": (1024, 2048),
        # 1024x2048 전체를 배치로 학습할 수 없으므로 고정 크기 크롭을 쓴다.
        # 640은 SegFormer 의 stride 32 에 맞아떨어져 stage 해상도가 어긋나지 않는다.
        "train_crop": (640, 640),
        "batch_size": 4,
        # 225 % 4 = 1. 매 epoch 마지막 step이 배치 1로 돌면 BatchNorm 통계가
        # 한 장에서 나와 무의미해진다(디코더 헤드, mmseg baseline 전부 BN).
        # shuffle 때문에 버려지는 이미지는 epoch마다 달라져 데이터 손실은 없다.
        "drop_last": True,
        "ignore_index": 19,
        "class_names": _MINICITY_CLASS_NAMES,
        "class_colors": np.array([
            [128, 64, 128],   # Road
            [244, 35, 232],   # Sidewalk
            [70, 70, 70],     # Building
            [102, 102, 156],  # Wall
            [190, 153, 153],  # Fence
            [153, 153, 153],  # Pole
            [250, 170, 30],   # TrafficLight
            [220, 220, 0],    # TrafficSign
            [107, 142, 35],   # Vegetation
            [152, 251, 152],  # Terrain
            [70, 130, 180],   # Sky
            [220, 20, 60],    # Person
            [255, 0, 0],      # Rider
            [0, 0, 142],      # Car
            [0, 0, 70],       # Truck
            [0, 60, 100],     # Bus
            [0, 80, 100],     # Train
            [0, 0, 230],      # Motorcycle
            [119, 11, 32],    # Bicycle
            [0, 0, 0],        # Void
        ], dtype=np.uint8),
    },
}

if DATASET not in _DATASET_PROFILES:
    raise ValueError(
        f"Unknown SWEEP_DATASET={DATASET!r}. "
        f"Available: {sorted(_DATASET_PROFILES)}"
    )
_PROFILE = _DATASET_PROFILES[DATASET]

# 데이터와 결과 경로는 환경변수로 지정한다. 지정하지 않으면 저장소 기준
# 상대 경로(./datasets, ./results)를 사용한다.
#   DATA_ROOT   : 아래 구조를 갖는 디렉터리
#                   <DATA_ROOT>/<hr_dirname>/<A_set|B_set>/{train,val,test}
#                   <DATA_ROOT>/<lr_dirname>/<A_set|B_set>/{train,val,test}
#   RESULT_ROOT : 학습 결과와 checkpoint 가 저장될 디렉터리
_DATA_ROOT = Path(_env_str("SWEEP_DATA_ROOT", "./datasets"))
_RESULT_ROOT = Path(_env_str("SWEEP_RESULT_BASE_DIR", "./results"))

DATA_HR_DIR = Path(_env_str("SWEEP_DATA_HR_DIR",
                            str(_DATA_ROOT / _PROFILE["hr_dirname"] / FOLD)))
DATA_DIR = Path(_env_str("SWEEP_DATA_DIR",
                         str(_DATA_ROOT / _PROFILE["lr_dirname"] / FOLD)))
BASE_DIR = _RESULT_ROOT

# DGT-Net teacher checkpoint. student distillation 에서만 사용한다.
TEACHER_CKPT = Path(_env_str("SWEEP_TEACHER_CKPT",
                             str(BASE_DIR / 'teacher' / 'best_model.pth')))

TEACHER_CKPT_ORIGINAL = Path(_env_str("SWEEP_TEACHER_CKPT_ORIGINAL", str(TEACHER_CKPT)))
TEACHER_CKPT_REMAP_APPLIED_UPSTREAM = _env_bool("SWEEP_TEACHER_CKPT_REMAP_APPLIED", False)
TEACHER_CKPT_REMAP_SOURCE_UPSTREAM = _env_str("SWEEP_TEACHER_CKPT_REMAP_SOURCE", "")
TEACHER_CKPT_REMAP_DETAIL_UPSTREAM = _env_str("SWEEP_TEACHER_CKPT_REMAP_DETAIL", "")

# ──────────────────────────────────────────────────────────────────
# 1. GENERAL: 프로젝트 전반 및 실험 관리 설정
# ──────────────────────────────────────────────────────────────────
class GENERAL:
    # 실험 프로젝트 이름
    PROJECT_NAME = _env_str("SWEEP_PROJECT_NAME", f"{DATASET}_{FOLD}")

    # 결과 파일을 저장할 기본 경로
    BASE_DIR = BASE_DIR / PROJECT_NAME
    LOG_DIR = BASE_DIR / "log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    SUMMARY_TXT = LOG_DIR / "training_summary.txt"
    SAVE_PLOT = LOG_DIR / "training_progress.png"

    SEED = 42

# ──────────────────────────────────────────────────────────────────
# 2. DATA: 데이터셋 관련 설정
# ──────────────────────────────────────────────────────────────────
class DATA:
    # 데이터셋 경로
    DATA_HR_DIR   = DATA_HR_DIR  # data_loader에서 사용하던 경로
    TRAIN_HR_DIR = DATA_HR_DIR / "train"
    VAL_HR_DIR = DATA_HR_DIR / "val"
    TEST_HR_DIR = DATA_HR_DIR / "test"

    DATA_DIR   = DATA_DIR  # data_loader에서 사용하던 경로
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"

    # HR 교사 이미지와 pseudo-depth는 둘 다 privileged 입력이라
    # 파일 상단의 USE_PRIVILEGED_INPUTS 하나로 같이 켜고 끈다.
    TRAIN_DEPTH_DIR = TRAIN_HR_DIR / "depths" if USE_PRIVILEGED_INPUTS else None
    VAL_DEPTH_DIR = VAL_HR_DIR / "depths" if USE_PRIVILEGED_INPUTS else None
    TEST_DEPTH_DIR = TEST_HR_DIR / "depths" if USE_PRIVILEGED_INPUTS else None

    TRAIN_IMG_DIR = TRAIN_DIR / "images"
    # teacher(HR) 전용 오프라인 클린 이미지
    TRAIN_TEACHER_IMG_DIR = TRAIN_HR_DIR / "images" if USE_PRIVILEGED_INPUTS else None
    TRAIN_LABEL_DIR = TRAIN_DIR / "labels"
    VAL_IMG_DIR = VAL_DIR / "images"
    VAL_TEACHER_IMG_DIR = VAL_HR_DIR / "images" if USE_PRIVILEGED_INPUTS else None
    VAL_LABEL_DIR = VAL_DIR / "labels"
    TEST_IMG_DIR = TEST_DIR / "images"
    TEST_TEACHER_IMG_DIR = TEST_HR_DIR / "images" if USE_PRIVILEGED_INPUTS else None
    TEST_LABEL_DIR = TEST_DIR / "labels"

    FILE_LIST = None

    # 어떤 프로파일로 돌고 있는지 (로그/결과 파일에 남기기 위한 값)
    NAME = DATASET

    # 입력 이미지 해상도 >> 원본 이미지의 크기가 아닌 모델에 들어가게 되는 input size
    # CamVid (360, 480) / MiniCity (1024, 2048)
    INPUT_RESOLUTION = tuple(_env_json("SWEEP_INPUT_RESOLUTION", list(_PROFILE["input_resolution"])))  # H, W

    # 학습 시 랜덤 크롭 크기 (H, W).
    # None이면 INPUT_RESOLUTION 전체를 쓰고 기존 CamVid 방식(80~100% 스케일 크롭 후
    # 원래 크기로 복원)이 적용된다. 값이 있으면 그 크기로 고정 크롭하고 리사이즈하지
    # 않는다 - 배치 내 텐서 크기가 같아야 하므로 이 경우 크롭은 항상 수행된다.
    TRAIN_CROP = _env_json("SWEEP_TRAIN_CROP", _PROFILE["train_crop"])
    if TRAIN_CROP is not None:
        TRAIN_CROP = tuple(TRAIN_CROP)

    # 배치 사이즈 및 데이터 로딩 워커 수
    BATCH_SIZE = _env_int("SWEEP_BATCH_SIZE", _PROFILE["batch_size"])

    # 학습 로더에서 크기가 모자란 마지막 배치를 버릴지 (val/test는 항상 유지).
    DROP_LAST = bool(_PROFILE["drop_last"])

    # 클래스 정보 (마지막 클래스가 Void = IGNORE_INDEX)
    CLASS_NAMES = list(_PROFILE["class_names"])
    NUM_CLASSES = len(CLASS_NAMES)  # CamVid=12, MiniCity=20
    IGNORE_INDEX = int(_PROFILE["ignore_index"])  # 'Void' 클래스의 인덱스

    # grayscale label(ground truth 포함)을 공식 컬러 매핑과 동일하게 시각화를 위해 컬러 매핑
    CLASS_COLORS = _PROFILE["class_colors"]

    assert IGNORE_INDEX == NUM_CLASSES - 1, (
        f"{DATASET}: Void는 마지막 클래스여야 한다 "
        f"(ignore_index={IGNORE_INDEX}, num_classes={NUM_CLASSES})"
    )
    assert len(CLASS_COLORS) == NUM_CLASSES

# ──────────────────────────────────────────────────────────────────
# 3. MODEL: 모델 설정
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# 4. TRAIN: 훈련 과정 관련 설정
# ──────────────────────────────────────────────────────────────────
class TRAIN:
    EPOCHS = _env_int("SWEEP_TRAIN_EPOCHS", 150)
    USE_AMP = False
    ACCUM_STEPS = 1
    GRAD_CLIP_NORM = 1.0

    # main문 실행시 checkpoint 로드할것인지 설정
    USE_CHECKPOINT = False
    CHECKPOINT_DIR = Path(_env_str("SWEEP_CHECKPOINT_DIR", "./results/resume"))

    FINETUNE = False
    FINETUNE_WEIGHT_DIR = _env_str("SWEEP_FINETUNE_WEIGHT", "./results/finetune/best_model.pth")
    # 딕셔너리 형태로 통일
    OPTIMIZER = {
        "NAME": "AdamW",
        "PARAMS": {
            "lr": 6e-5,
            "weight_decay": 5e-3
        }
    }
    # Param-group 별 lr / weight_decay (train_kd.py 에서 사용).
    # AdamW 의 betas, eps 등은 OPTIMIZER["PARAMS"] 로 제어한다.
    PARAM_GROUPS = {
        "student": {"lr": 6e-5, "weight_decay": 5e-3},
        "kd_extra": {"lr": 3e-4, "weight_decay": 0.0},
        # teacher는 freeze=False이고 teacher CE를 쓰는 경우에만 optimizer에 포함
        "teacher": {"lr": 6e-5, "weight_decay": 5e-3},
    }

    SCHEDULER_RoP = {
        "NAME": "ReduceLROnPlateau",
        "PARAMS": {
            "mode": 'min',
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6
        }
    }

    LOSS_FN = {
        "NAME": "CrossEntropyLoss",
        "PARAMS": {
            "ignore_index": DATA.IGNORE_INDEX
        }
    }

# ──────────────────────────────────────────────────────────────────
# 5. KD: Knowledge Distillation 관련 설정
# ──────────────────────────────────────────────────────────────────
class KD:
    ENABLE = True

    ENGINE_NAME = _env_str("SWEEP_ENGINE_NAME", "fdcs_gbst_teacher")
    """
    available engines:
    fdcs_gbst_teacher                teacher training
    priv_reach_geo_dec_stagek_sgsc   student distillation
    """

    # 모델 선택. 사용 가능한 이름은 두 가지뿐이다.
    #   segformerb0        DS-Net (student)
    #   segformerb3_fdcs   DGT-Net (teacher, FDM + LCFC)
    TEACHER_NAME = _env_str("SWEEP_TEACHER_NAME", "segformerb3_fdcs")
    STUDENT_NAME = _env_str("SWEEP_STUDENT_NAME", "segformerb3_fdcs")
    # 교사 고정 여부
    FREEZE_TEACHER = _env_bool("SWEEP_FREEZE_TEACHER", False)

    ALL_ENGINE_PARAMS = {
        "fdcs_gbst_teacher": {
            "w_ce_hr": _env_float("SWEEP_FDCS_GBST_W_CE_HR", 0.5),
            "w_ce_lr": _env_float("SWEEP_FDCS_GBST_W_CE_LR", 1.0),
            "w_lapc": _env_float("SWEEP_FDCS_GBST_W_LAPC", 0.2),
            "eval_depth_mode": _env_str("SWEEP_FDCS_GBST_EVAL_DEPTH_MODE", "zero"),
            "lapc_detach_lr": _env_bool("SWEEP_FDCS_GBST_LAPC_DETACH_LR", True),
            "num_classes": DATA.NUM_CLASSES,
            "ignore_index": DATA.IGNORE_INDEX,
        },
        "priv_reach_geo_dec_stagek_sgsc": {
            "w_ce_student": 1.0,
            "lambda_stage": _env_float(
                "SWEEP_PRIV_REACH_GEO_STAGEK_LAMBDA_STAGE",
                _env_float("SWEEP_PRIV_REACH_GEO_DEC_LAMBDA_STAGE", 1.75),
            ),
            "lambda_decoder": _env_float(
                "SWEEP_PRIV_REACH_GEO_STAGEK_LAMBDA_DECODER",
                _env_float("SWEEP_PRIV_REACH_GEO_DEC_LAMBDA_DECODER", 0.75),
            ),
            "k_decoder": _env_int(
                "SWEEP_PRIV_REACH_GEO_STAGEK_K_DECODER",
                _env_int("SWEEP_PRIV_REACH_GEO_DEC_K_DECODER", 32),
            ),
            # Encoder stage-wise fixed k. Defaults match the original fixed-k
            # engine; set S1/S2/S3 independently for stage-wise sweeps.
            "k_stage": _env_int(
                "SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE",
                _env_int("SWEEP_PRIV_REACH_GEO_DEC_K_STAGE", 64),
            ),
            "k_stage_by_stage": {
                0: _env_int(
                    "SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE_0",
                    _env_int("SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE", 64),
                ),
                1: _env_int(
                    "SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE_1",
                    _env_int("SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE", 64),
                ),
                2: _env_int(
                    "SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE_2",
                    _env_int("SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE", 64),
                ),
                3: _env_int(
                    "SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE_3",
                    _env_int("SWEEP_PRIV_REACH_GEO_STAGEK_K_STAGE", 64),
                ),
            },
            "apply_stages": _env_str(
                "SWEEP_PRIV_REACH_GEO_STAGEK_APPLY_STAGES",
                _env_str("SWEEP_PRIV_REACH_GEO_DEC_APPLY_STAGES", "1, 2, 3"),
            ),
            "stage_weights": _env_json(
                "SWEEP_PRIV_REACH_GEO_STAGEK_STAGE_WEIGHTS",
                {0: 0.0, 1: 0.5, 2: 1.5, 3: 1.0},
            ),
            "student_channels": [32, 64, 160, 256],
            "teacher_channels": [64, 128, 320, 512],
            "student_dec_ch": _env_int("SWEEP_PRIV_REACH_GEO_STAGEK_STUDENT_DEC_CH", 256),
            "teacher_dec_ch": _env_int("SWEEP_PRIV_REACH_GEO_STAGEK_TEACHER_DEC_CH", 768),
            "decoder_teacher_view": _env_str("SWEEP_PRIV_REACH_GEO_STAGEK_TEACHER_VIEW", "hr"),
            "decoder_depth_mode": _env_str("SWEEP_PRIV_REACH_GEO_STAGEK_DEPTH_MODE", "input"),
            "ignore_index": DATA.IGNORE_INDEX,
            "rg_csf_hidden_ch": _env_int("SWEEP_PRIV_REACH_GEO_STAGEK_CSF_HIDDEN_CH", 64),
            "rg_csf_reduction": _env_int("SWEEP_PRIV_REACH_GEO_STAGEK_CSF_REDUCTION", 4),
            "rg_lr_scale_alpha": _env_float("SWEEP_PRIV_REACH_GEO_STAGEK_LR_SCALE_ALPHA", 1.0),
            "rg_depth_gate_alpha": _env_float("SWEEP_PRIV_REACH_GEO_STAGEK_DEPTH_GATE_ALPHA", 1.0),
            "rg_depth_similarity_power": _env_float("SWEEP_PRIV_REACH_GEO_STAGEK_DEPTH_SIM_POWER", 1.0),
            "rg_weight_min": _env_float("SWEEP_PRIV_REACH_GEO_STAGEK_WEIGHT_MIN", 0.8),
            "rg_weight_max": _env_float("SWEEP_PRIV_REACH_GEO_STAGEK_WEIGHT_MAX", 1.2),
        },
    }

    ENGINE_PARAMS = ALL_ENGINE_PARAMS[ENGINE_NAME]
