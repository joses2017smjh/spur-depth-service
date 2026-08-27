"""TensorRT 10+ dropped EXPLICIT_BATCH. Do not import the real wheel here."""

from __future__ import annotations

from spur_depth.export.trt_runtime import builder_has_fp16, enable_fp16, network_creation_flags


class _Trt8:
    class NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 0


class _Trt11:
    class NetworkDefinitionCreationFlag:
        pass


class _TrtFuture:
    pass


def test_network_flags_legacy_explicit_batch():
    assert network_creation_flags(_Trt8) == 1


def test_network_flags_trt11_default_explicit():
    assert network_creation_flags(_Trt11) == 0
    assert network_creation_flags(_TrtFuture) == 0


def test_builder_has_fp16_when_attr_missing():
    class _Trt11Builder:
        pass

    class _Trt8Builder:
        platform_has_fast_fp16 = True

    assert builder_has_fp16(_Trt11Builder()) is True
    assert builder_has_fp16(_Trt8Builder()) is True


def test_enable_fp16_skips_when_flag_removed():
    class _Flag:
        pass

    class _Trt11:
        BuilderFlag = _Flag

    class _Cfg:
        def __init__(self):
            self.n = 0

        def set_flag(self, _flag):
            self.n += 1

    cfg = _Cfg()
    assert enable_fp16(_Trt11, cfg) is False
    assert cfg.n == 0
