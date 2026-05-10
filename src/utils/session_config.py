"""
会话级配置管理：读取、修改、保存 config.json
支持属性式访问和字典式访问
"""
import json
import os
from config import get_dataset_dir


class SessionConfig(dict):
    """
    可持久化的配置字典，支持属性式访问和 .save() 方法。

    用法示例:
        config = SessionConfig.from_dataset('BCICIV_2a', 'A01', 'T')
        print(config.resample_freq)          # 属性访问
        config['ica_exclude'] = [0, 2, 6]    # 字典修改
        config.save()                        # 保存到 config.json
    """

    def __init__(self, dataset_name: str, subject_id: str, session: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset_name = dataset_name
        self._subject_id = subject_id
        self._session = session

    # ---------- 属性式访问 ----------
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'SessionConfig' 对象没有属性 '{key}'")

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self[key] = value

    # ---------- 保存方法 ----------
    def save(self):
        """
        将当前配置保存到 config.json 中对应被试/会话的条目下。
        只保存与 default 不同的参数（自动 diff）。
        """
        config_path = os.path.join(get_dataset_dir(self._dataset_name), 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r') as f:
            all_cfg = json.load(f)

        defaults = all_cfg.get('default', {})

        # 计算与 default 的差异
        diff = {}
        for key, value in self.items():
            if value is None:          # 可以按需删除此判断以保存 null
                continue
            if key in defaults:
                if value != defaults[key]:
                    diff[key] = value
            else:
                diff[key] = value

        # 更新被试/会话条目
        if self._subject_id not in all_cfg:
            all_cfg[self._subject_id] = {}
        all_cfg[self._subject_id][self._session] = diff

        # 写回文件
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(all_cfg, f, indent=2, ensure_ascii=False)
        print(f"✓ 配置已保存到 {config_path}")

    @classmethod
    def from_dataset(cls, dataset_name: str, subject_id: str, session: str) -> 'SessionConfig':
        """
        从 config.json 加载配置，合并 default + 被试/会话覆盖，返回可修改的 SessionConfig 实例。
        """
        config_path = os.path.join(get_dataset_dir(dataset_name), 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, 'r') as f:
            all_cfg = json.load(f)

        params = all_cfg.get('default', {}).copy()
        overrides = all_cfg.get(subject_id, {}).get(session, {})
        params.update(overrides)

        return cls(dataset_name, subject_id, session, params)