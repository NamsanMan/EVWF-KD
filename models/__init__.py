from .segformer_wrapper import SegFormerWrapper
from .segformer_fdcs_teacher_wrapper import SegFormerFDCSTeacherWrapper

import config


def create_model(model_name: str):
    """
    논문에서 사용하는 두 model 만 생성한다.
      segformerb0        : DS-Net (student)
      segformerb3_fdcs   : DGT-Net (teacher, FDM + LCFC)
    """
    model_name = model_name.lower()
    num_classes = config.DATA.NUM_CLASSES

    segformer_names = {"segformerb0", "segformerb3"}

    if model_name.endswith("_fdcs"):
        base = model_name[:-5]
        if base not in segformer_names:
            raise ValueError(
                f"FDCS teacher wrapper is only supported for SegFormer models, got '{model_name}'."
            )
        model = SegFormerFDCSTeacherWrapper(base)
        print(f"Model '{model_name}' created (DGT-Net teacher).", flush=True)

    elif model_name in segformer_names:
        model = SegFormerWrapper(model_name)
        print(f"Model '{model_name}' created.", flush=True)

    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    print(f"  - Number of classes: {num_classes}", flush=True)
    print(f"  - IGNORE_INDEX     : {config.DATA.IGNORE_INDEX}", flush=True)
    return model
