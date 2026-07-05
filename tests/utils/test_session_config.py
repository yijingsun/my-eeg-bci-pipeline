"""
SessionConfig 单元测试
覆盖: 从 config.json 加载 → 合并默认值与被试覆盖 → 属性/字典访问 → 保存(diff)
"""
import json
import os

import pytest

from src.utils.session_config import SessionConfig


# ============================================================
# from_json_file — 加载与合并
# ============================================================

class TestFromJsonFile:
    """测试 from_json_file() 加载真实/模拟 config.json"""

    def test_load_defaults_only(self, real_config_path):
        """A01/T 无被试覆盖 → 全部使用默认值"""
        cfg = SessionConfig.from_json_file(real_config_path, "A01", "T")
        assert cfg["tmin"] == 1.0
        assert cfg["tmax"] == 4.0
        assert cfg["resample_freq"] is None
        assert cfg["ref_type"] == "average"
        assert cfg["csp_n_components"] == 6
        assert cfg["expected_trials"] == 288

    def test_load_with_overrides(self, real_config_path):
        """A02/T 有被试级 tmin/tmax 覆盖"""
        cfg = SessionConfig.from_json_file(real_config_path, "A02", "T")
        assert cfg["tmin"] == 2.0    # 覆盖
        assert cfg["tmax"] == 5.0    # 覆盖
        assert cfg["resample_freq"] is None  # 未被覆盖，来自 default

    def test_load_mixed_override(self, real_config_path):
        """A04/T 同时覆盖了标量和 dict（mi_event_mapping）"""
        cfg = SessionConfig.from_json_file(real_config_path, "A04", "T")
        assert cfg["tmin"] == 1.5
        assert cfg["tmax"] == 3.5
        assert cfg["mi_event_mapping"] == {"769": 5, "770": 6, "771": 7, "772": 8}
        # 未覆盖的仍来自 default
        assert cfg["csp_n_components"] == 6

    def test_load_evaluation_session(self, real_config_path):
        """A01/E 有 E 会话覆盖（tmin=3.0, tmax=6.0）"""
        cfg = SessionConfig.from_json_file(real_config_path, "A01", "E")
        assert cfg["tmin"] == 3.0
        assert cfg["tmax"] == 6.0

    def test_dataset_name_inferred(self, real_config_path):
        """从路径自动推断 dataset_name"""
        cfg = SessionConfig.from_json_file(real_config_path, "A01", "T")
        assert cfg._dataset_name == "BCICIV_2a"

    def test_config_path_stored(self, real_config_path):
        """_config_path 应正确保存"""
        cfg = SessionConfig.from_json_file(real_config_path, "A01", "T")
        assert cfg._config_path == real_config_path

    # --- 模拟 config 测试 ---

    def test_minimal_config_default(self, minimal_config_path):
        cfg = SessionConfig.from_json_file(minimal_config_path, "S02", "T")
        assert cfg["a"] == 1
        assert cfg["b"] == "hello"

    def test_minimal_config_override(self, minimal_config_path):
        cfg = SessionConfig.from_json_file(minimal_config_path, "S01", "T")
        assert cfg["a"] == 99    # 被试覆盖
        assert cfg["b"] == "hello"  # 来自 default

    def test_minimal_config_empty_override(self, minimal_config_path):
        """E 会话无覆盖 → 全用 default"""
        cfg = SessionConfig.from_json_file(minimal_config_path, "S01", "E")
        assert cfg["a"] == 1
        assert cfg["b"] == "hello"

    # --- 错误路径 ---

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            SessionConfig.from_json_file("/nonexistent/config.json", "X", "T")


# ============================================================
# 属性访问 & 字典访问
# ============================================================

class TestAccess:
    """测试属性式访问和字典式访问"""

    @pytest.fixture
    def cfg(self, real_config_path):
        return SessionConfig.from_json_file(real_config_path, "A01", "T")

    def test_attribute_read(self, cfg):
        assert cfg.tmin == 1.0
        assert cfg.tmax == 4.0
        assert cfg.resample_freq is None

    def test_dict_read(self, cfg):
        assert cfg["tmin"] == 1.0
        assert cfg.get("nonexistent", 42) == 42

    def test_attribute_write(self, cfg):
        cfg.new_key = "new_value"
        assert cfg["new_key"] == "new_value"

    def test_dict_write(self, cfg):
        cfg["new_key"] = 123
        assert cfg.new_key == 123

    def test_unknown_attribute_raises(self, cfg):
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_key

    def test_private_attr_does_not_go_to_dict(self, cfg):
        """以 _ 开头的属性存储为实例属性，不进入字典"""
        cfg._private = "secret"
        assert "_private" not in cfg
        assert cfg._private == "secret"


# ============================================================
# 构造方式
# ============================================================

class TestConstructors:

    def test_direct_constructor(self):
        cfg = SessionConfig("TestDS", "S01", "T", {"a": 1, "b": 2})
        assert cfg["a"] == 1
        assert cfg._dataset_name == "TestDS"
        assert cfg._subject_id == "S01"
        assert cfg._session == "T"

    def test_direct_constructor_with_config_path(self):
        cfg = SessionConfig("DS", "S01", "T", {"x": 1},
                            config_path="/tmp/config.json")
        assert cfg._config_path == "/tmp/config.json"

    def test_constructor_without_config_path_cannot_save(self):
        cfg = SessionConfig("DS", "S01", "T", {"x": 1})
        with pytest.raises(RuntimeError, match="未设置 config 文件路径"):
            cfg.save()


# ============================================================
# save — diff 保存
# ============================================================

class TestSave:
    """测试 save() 的 diff 逻辑：仅保存与 default 不同的参数"""

    def test_save_only_diff(self, temp_config_path):
        """修改 default 值后保存 → 仅保存差异项"""
        cfg = SessionConfig.from_json_file(temp_config_path, "A01", "T")
        original_tmin = cfg["tmin"]

        # 修改 tmin（与 default 不同）
        cfg["tmin"] = original_tmin + 99
        cfg["new_param"] = "added"
        cfg.save()

        # 重新加载验证
        reloaded = SessionConfig.from_json_file(temp_config_path, "A01", "T")
        assert reloaded["tmin"] == original_tmin + 99
        assert reloaded["new_param"] == "added"
        # 未修改的参数应保持 default
        assert reloaded["resample_freq"] is None

    def test_save_skips_null_values(self, temp_config_path):
        """None 值不会被写入 diff"""
        cfg = SessionConfig.from_json_file(temp_config_path, "A01", "T")
        cfg["resample_freq"] = None   # default 就是 None → 不写入
        cfg.save()

        # 重新加载 — 不应有额外条目
        with open(temp_config_path) as f:
            saved = json.load(f)
        a01_t = saved.get("A01", {}).get("T", {})
        assert "resample_freq" not in a01_t

    def test_save_creates_subject_entry_if_missing(self, temp_config_path):
        """为配置中不存在的被试创建条目"""
        cfg = SessionConfig.from_json_file(temp_config_path, "A99", "T")
        cfg["tmin"] = 5.0
        cfg.save()

        reloaded = SessionConfig.from_json_file(temp_config_path, "A99", "T")
        assert reloaded["tmin"] == 5.0

    def test_save_preserves_other_subjects(self, temp_config_path):
        """save 不应覆盖其他被试的配置"""
        cfg = SessionConfig.from_json_file(temp_config_path, "A01", "T")
        cfg["tmin"] = 9.9
        cfg.save()

        # A02/T 应保持不变
        reloaded_a02 = SessionConfig.from_json_file(temp_config_path, "A02", "T")
        assert reloaded_a02["tmin"] == 2.0


# ============================================================
# from_dataset — 集成
# ============================================================

class TestFromDataset:
    """测试 from_dataset() 向后兼容方法"""

    def test_from_dataset_returns_same_as_from_json_file(self, real_config_path):
        cfg1 = SessionConfig.from_dataset("BCICIV_2a", "A02", "T")
        cfg2 = SessionConfig.from_json_file(real_config_path, "A02", "T")
        assert cfg1["tmin"] == cfg2["tmin"]
        assert cfg1["tmax"] == cfg2["tmax"]
        assert cfg1._dataset_name == cfg2._dataset_name
