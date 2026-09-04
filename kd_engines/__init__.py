from .base_engine import BaseKDEngine
from .fdcs_gbst_teacher import FDCSGBSTTeacherEngine
from .privileged_reachable_geo_dec_stagek_sgsc import (
    PrivilegedReachableGeoDecStageKSGSCEngine,
)

# 논문에서 사용하는 두 엔진만 등록한다.
#   fdcs_gbst_teacher              : teacher 학습 엔진
#   priv_reach_geo_dec_stagek_sgsc : student distillation 엔진
KD_ENGINE_REGISTRY = {
    "fdcs_gbst_teacher": FDCSGBSTTeacherEngine,
    "priv_reach_geo_dec_stagek_sgsc": PrivilegedReachableGeoDecStageKSGSCEngine,
}


def create_kd_engine(config, teacher, student):
    """
    config 파일의 내용을 바탕으로 적절한 KD 엔진 객체를 생성하여 반환합니다.
    """
    engine_name = config.ENGINE_NAME

    if hasattr(config, "ALL_ENGINE_PARAMS"):
        engine_params = config.ALL_ENGINE_PARAMS.get(engine_name, {})
    else:
        engine_params = getattr(config, "ENGINE_PARAMS", {})

    if engine_name not in KD_ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown KD Engine: {engine_name}. "
            f"Available engines: {list(KD_ENGINE_REGISTRY.keys())}"
        )

    engine_class = KD_ENGINE_REGISTRY[engine_name]

    # teacher, student 모델과 함께 파라미터를 전달하여 엔진 객체 생성
    kd_engine = engine_class(teacher=teacher, student=student, **engine_params)

    return kd_engine
