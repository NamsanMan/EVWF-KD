import os
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as T

import config

# Dateset 구현
class CamVidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list=None, transform=None, teacher_images_dir=None, depth_dir=None):
        self.images_dir = images_dir
        self.teacher_images_dir = teacher_images_dir
        self.depth_dir = depth_dir
        self.masks_dir = masks_dir
        if file_list:
            with open(file_list) as f:
                self.files = [line.strip() for line in f]
        else:
            self.files = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        teacher_image = None
        teacher_path = None
        depth_image = None
        if self.teacher_images_dir is not None:
            teacher_path = os.path.join(self.teacher_images_dir, filename)
            if not os.path.exists(teacher_path):
                raise FileNotFoundError(f"Teacher HR image not found: {teacher_path}")
            teacher_image = Image.open(teacher_path).convert("RGB")

        if self.depth_dir is not None:
            depth_path = os.path.join(self.depth_dir, filename)
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth image not found: {depth_path}")
            # depth는 photometric jitter 대상이 아니므로 원본 단일 채널로 유지
            depth_image = Image.open(depth_path)

        if self.transform:
            if teacher_image is not None and depth_image is not None:
                image, mask, teacher_image, depth_image = self.transform(
                    image, mask, teacher_image, depth_image
                )
            elif teacher_image is not None:
                image, mask, teacher_image = self.transform(image, mask, teacher_image)
            elif depth_image is not None:
                image, mask, depth_image = self.transform(image, mask, depth=depth_image)
            else:
                image, mask = self.transform(image, mask)

        if teacher_image is not None and depth_image is not None:
            return (image, teacher_image, depth_image), mask
        if teacher_image is not None:
            return (image, teacher_image), mask
        if depth_image is not None:
            return (image, depth_image), mask

        return image, mask

# train set에 대한 data augmentation: random crop, random flip, color jitter
def _resolve_crop_size(crop_size, size):
    """학습 크롭 크기를 검증한다. None이면 크롭 없이 전체 해상도를 쓴다.

    크롭은 초기 리사이즈(=size) 뒤에 적용되므로 size보다 클 수 없다.
    조용히 잘리면 배치 안에서 크기가 어긋나므로 여기서 바로 막는다.
    """
    if crop_size is None:
        return None
    crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
    if crop_h > size[0] or crop_w > size[1]:
        raise ValueError(
            f"crop_size {(crop_h, crop_w)} exceeds input resolution {tuple(size)}"
        )
    return (crop_h, crop_w)


class TrainAugmentation:
    def __init__(
        self,
        size,
        crop_size=None,
        hflip_prob: float = 0.5,
        crop_prob: float = 0.7,
        crop_range: tuple[float, float] = (80.0, 100.0),
        brightness: tuple[float, float] = (0.6, 1.4),
        contrast: tuple[float, float]   = (0.7, 1.2),
        saturation: tuple[float, float] = (0.9, 1.3),
        hue: tuple[float, float]        = (-0.05, 0.05),
    ):
        self.size = size
        self.crop_size = _resolve_crop_size(crop_size, size)
        self.hflip_prob = hflip_prob

        self.crop_prob = crop_prob
        self.crop_min, self.crop_max = crop_range

        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self, img, mask, depth=None):
        # 0) 초기 리사이즈
        img  = F.resize(img,  self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        if depth is not None:
            depth = F.resize(depth, self.size, interpolation=InterpolationMode.BILINEAR)

        # 1) 랜덤 크롭
        if self.crop_size is not None:
            # 고정 크기 크롭 (MiniCity 768x768). 배치 내 텐서 크기가 같아야 하므로
            # 확률로 건너뛰지 않고, 원래 크기로 되돌리지도 않는다.
            i, j, h, w = T.RandomCrop.get_params(img, output_size=self.crop_size)
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            if depth is not None:
                depth = F.crop(depth, i, j, h, w)
        elif random.random() < self.crop_prob:
            # 예: 원본의 80~100% 영역을 무작위 크롭 후 원래 크기로 리사이즈
            target_h, target_w = self.size
            scale_min = self.crop_min / 100.0
            scale_max = self.crop_max / 100.0
            crop_h = int(random.uniform(scale_min, scale_max) * target_h)
            crop_w = int(random.uniform(scale_min, scale_max) * target_w)
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(crop_h, crop_w))
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            if depth is not None:
                depth = F.crop(depth, i, j, h, w)
            img = F.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
            if depth is not None:
                depth = F.resize(depth, self.size, interpolation=InterpolationMode.BILINEAR)

        # 2) 랜덤 수평 뒤집기
        if random.random() < self.hflip_prob:
            img  = F.hflip(img)
            mask = F.hflip(mask)
            if depth is not None:
                depth = F.hflip(depth)

        # 4) 컬러 지터 (depth에는 적용하지 않음)
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        s = random.uniform(*self.saturation)
        h = random.uniform(*self.hue)
        img = F.adjust_brightness(img, b)
        img = F.adjust_contrast(img,   c)
        img = F.adjust_saturation(img, s)
        img = F.adjust_hue(img,        h)

        # 5) 텐서 변환 & 정규화
        img  = F.to_tensor(img)
        img  = F.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        # ★ 라벨 정규화: [0..10, 11] 이외는 11(Void)로 치환
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[(mask_np < 0) | (mask_np > config.DATA.IGNORE_INDEX)] = config.DATA.IGNORE_INDEX
        mask = torch.from_numpy(mask_np).long()

        if depth is not None:
            depth = np.array(depth, dtype=np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            lo = np.percentile(depth, 1.0)
            hi = np.percentile(depth, 99.0)
            if hi > lo + 1e-12:
                depth = np.clip(depth, lo, hi)
                depth = (depth - lo) / (hi - lo + 1e-12)
            else:
                depth = np.zeros_like(depth, dtype=np.float32)
            depth = torch.from_numpy(depth).unsqueeze(0).float()
            return img, mask, depth

        return img, mask


# 이미지와 마스크(레이블)을 동시에 전처리하기 위해 만든다
class SegmentationTransform:
    def __init__(self, size):
        # 크기(사이즈)
        self.size = size
    def __call__(self, img, mask, img_teacher=None, depth=None):
        img = F.resize(img, self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        # ★ 라벨 정규화: [0..10, 11] 이외는 11(Void)로 치환
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[(mask_np < 0) | (mask_np > config.DATA.IGNORE_INDEX)] = config.DATA.IGNORE_INDEX
        mask = torch.from_numpy(mask_np).long()
        if img_teacher is not None:
            img_teacher = F.resize(img_teacher, self.size)
            img_teacher = F.to_tensor(img_teacher)
            img_teacher = F.normalize(
                img_teacher,
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225],
            )

        if depth is not None:
            depth = F.resize(depth, self.size, interpolation=InterpolationMode.BILINEAR)
            depth = np.array(depth, dtype=np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            lo = np.percentile(depth, 1.0)
            hi = np.percentile(depth, 99.0)
            if hi > lo + 1e-12:
                depth = np.clip(depth, lo, hi)
                depth = (depth - lo) / (hi - lo + 1e-12)
            else:
                depth = np.zeros_like(depth, dtype=np.float32)
            depth = torch.from_numpy(depth).unsqueeze(0).float()

        if img_teacher is not None and depth is not None:
            return img, mask, img_teacher, depth
        if img_teacher is not None:
            return img, mask, img_teacher
        if depth is not None:
            return img, mask, depth
        return img, mask

class JointKDTrainAugmentation:
    """학생(LR)과 교사(HR) 이미지를 동일한 기하 변환으로 다루는 KD 전용 변환."""

    def __init__(
        self,
        size,
        crop_size=None,
        hflip_prob: float = 0.5,
        crop_prob: float = 0.7,
        crop_range: tuple[float, float] = (80.0, 100.0),
        brightness: tuple[float, float] = (0.6, 1.4),
        contrast: tuple[float, float]   = (0.7, 1.2),
        saturation: tuple[float, float] = (0.9, 1.3),
        hue: tuple[float, float]        = (-0.05, 0.05),
    ):
        self.size = size
        self.crop_size = _resolve_crop_size(crop_size, size)
        self.hflip_prob = hflip_prob
        self.crop_prob = crop_prob
        self.crop_min, self.crop_max = crop_range
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def _apply_same_color_jitter(self, img_student, img_teacher):
        """
        Apply identical photometric jitter to student and teacher images.
        This enforces identical input distribution for LR->LR KD.
        """
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        s = random.uniform(*self.saturation)
        h = random.uniform(*self.hue)

        img_student = F.adjust_brightness(img_student, b)
        img_teacher = F.adjust_brightness(img_teacher, b)

        img_student = F.adjust_contrast(img_student, c)
        img_teacher = F.adjust_contrast(img_teacher, c)

        img_student = F.adjust_saturation(img_student, s)
        img_teacher = F.adjust_saturation(img_teacher, s)

        img_student = F.adjust_hue(img_student, h)
        img_teacher = F.adjust_hue(img_teacher, h)

        return img_student, img_teacher

    def __call__(self, img_student, mask, img_teacher=None, depth=None):
        img_teacher = img_teacher if img_teacher is not None else img_student

        # 초기 리사이즈
        img_student = F.resize(img_student, self.size)
        img_teacher = F.resize(img_teacher, self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        if depth is not None:
            depth = F.resize(depth, self.size, interpolation=InterpolationMode.BILINEAR)

        # 동일 파라미터의 랜덤 크롭
        if self.crop_size is not None:
            # 고정 크기 크롭 (MiniCity 768x768). 학생/교사/라벨/깊이에 같은 박스를
            # 적용하므로 정렬이 유지된다. 배치 크기 일관성 때문에 항상 수행한다.
            i, j, h, w = T.RandomCrop.get_params(img_student, output_size=self.crop_size)
            img_student = F.crop(img_student, i, j, h, w)
            img_teacher = F.crop(img_teacher, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            if depth is not None:
                depth = F.crop(depth, i, j, h, w)
        elif random.random() < self.crop_prob:
            target_h, target_w = self.size
            scale_min = self.crop_min / 100.0
            scale_max = self.crop_max / 100.0
            crop_h = int(random.uniform(scale_min, scale_max) * target_h)
            crop_w = int(random.uniform(scale_min, scale_max) * target_w)
            i, j, h, w = T.RandomCrop.get_params(img_student, output_size=(crop_h, crop_w))
            img_student = F.crop(img_student, i, j, h, w)
            img_teacher = F.crop(img_teacher, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            if depth is not None:
                depth = F.crop(depth, i, j, h, w)
            img_student = F.resize(img_student, self.size, interpolation=InterpolationMode.BILINEAR)
            img_teacher = F.resize(img_teacher, self.size, interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
            if depth is not None:
                depth = F.resize(depth, self.size, interpolation=InterpolationMode.BILINEAR)

        # 랜덤 수평 뒤집기
        if random.random() < self.hflip_prob:
            img_student = F.hflip(img_student)
            img_teacher = F.hflip(img_teacher)
            mask = F.hflip(mask)
            if depth is not None:
                depth = F.hflip(depth)

        # 공통 photometric jitter: student/teacher 동일 분포 강제
        img_student, img_teacher = self._apply_same_color_jitter(img_student, img_teacher)

        # Tensor 및 정규화
        img_student = F.to_tensor(img_student)
        img_student = F.normalize(img_student, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        img_teacher = F.to_tensor(img_teacher)
        img_teacher = F.normalize(img_teacher, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        if depth is not None:
            depth = np.array(depth, dtype=np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            lo = np.percentile(depth, 1.0)
            hi = np.percentile(depth, 99.0)
            if hi > lo + 1e-12:
                depth = np.clip(depth, lo, hi)
                depth = (depth - lo) / (hi - lo + 1e-12)
            else:
                depth = np.zeros_like(depth, dtype=np.float32)
            depth = torch.from_numpy(depth).unsqueeze(0).float()

        mask_np = np.array(mask, dtype=np.int64)
        mask_np[(mask_np < 0) | (mask_np > config.DATA.IGNORE_INDEX)] = config.DATA.IGNORE_INDEX
        mask = torch.from_numpy(mask_np).long()

        if depth is not None:
            return img_student, mask, img_teacher, depth
        return img_student, mask, img_teacher


def _resolve_depth_dir(attr_name: str):
    """depth 디렉터리가 실제로 존재할 때만 반환, 아니면 None."""
    d = getattr(config.DATA, attr_name, None)
    if d is not None and os.path.isdir(d):
        return str(d)
    return None


_train_teacher_dir = config.DATA.TRAIN_TEACHER_IMG_DIR
if _train_teacher_dir is not None and os.path.exists(_train_teacher_dir):
    _train_transform = JointKDTrainAugmentation(
        size=config.DATA.INPUT_RESOLUTION,
        crop_size=config.DATA.TRAIN_CROP,
    )
else:
    _train_teacher_dir = None
    _train_transform = TrainAugmentation(
        size=config.DATA.INPUT_RESOLUTION,
        crop_size=config.DATA.TRAIN_CROP,
    )

# A_set 만 B_set으로 바꿔서 2fold 진행
train_dataset = CamVidDataset(
    images_dir = config.DATA.TRAIN_IMG_DIR,
    masks_dir  = config.DATA.TRAIN_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform = _train_transform,
    teacher_images_dir=_train_teacher_dir,
    depth_dir=_resolve_depth_dir("TRAIN_DEPTH_DIR"),
)

val_dataset = CamVidDataset(
    images_dir = config.DATA.VAL_IMG_DIR,
    masks_dir  = config.DATA.VAL_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform = SegmentationTransform(config.DATA.INPUT_RESOLUTION),
    depth_dir=_resolve_depth_dir("VAL_DEPTH_DIR"),
)

test_dataset = CamVidDataset(
    images_dir = config.DATA.TEST_IMG_DIR,
    masks_dir  = config.DATA.TEST_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform =SegmentationTransform(config.DATA.INPUT_RESOLUTION),
    depth_dir=_resolve_depth_dir("TEST_DEPTH_DIR"),
)

# 데이터셋을 train.py로 넘겨줌
# train_loader에만 shuffle true
# num_workers는 전부 같은 값으로 통일(0 아니면 1)
# val_loader와 test_lodaer의 batch_size는 1로 하는게 맞고, train_loader의 batchsize는 4를 추천
train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE,  shuffle=True,  num_workers=0, drop_last=config.DATA.DROP_LAST)
val_loader   = DataLoader(val_dataset,   batch_size=1,  shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=1,  shuffle=False, num_workers=0)