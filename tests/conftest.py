"""
共享测试夹具
"""
import json
import os
import shutil
import tempfile

import pytest

# 真实项目 config.json 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "data", "BCICIV_2a", "config.json"
)


@pytest.fixture(scope="session")
def real_config_path() -> str:
    """真实项目的 config.json 路径"""
    if not os.path.exists(REAL_CONFIG_PATH):
        pytest.skip(f"真实配置文件不存在: {REAL_CONFIG_PATH}")
    return REAL_CONFIG_PATH


@pytest.fixture
def temp_config_path(request):
    """创建临时 config.json 副本，测试结束自动删除。
    用于 save() 测试，避免污染真实文件。
    """
    with open(REAL_CONFIG_PATH, "r") as f:
        original = json.load(f)

    tmpdir = tempfile.mkdtemp(prefix="test_config_")
    tmp_path = os.path.join(tmpdir, "config.json")

    # 写入副本
    with open(tmp_path, "w") as f:
        json.dump(original, f, indent=2)

    yield tmp_path

    # 清理
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def minimal_config_path():
    """创建一个最简 config.json，用于基本功能测试"""
    tmpdir = tempfile.mkdtemp(prefix="test_minimal_")
    tmp_path = os.path.join(tmpdir, "config.json")

    data = {
        "default": {
            "a": 1,
            "b": "hello",
            "c": None,
        },
        "S01": {
            "T": {"a": 99},
            "E": {},
        },
    }
    with open(tmp_path, "w") as f:
        json.dump(data, f)

    yield tmp_path

    shutil.rmtree(tmpdir, ignore_errors=True)
