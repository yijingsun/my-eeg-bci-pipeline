"""
会话级配置管理：读取、修改、保存 config.json
支持属性式访问和字典式访问
"""
import json
import os


class SessionConfig(dict):
    """
    可持久化的配置字典，支持属性式访问和 .save() 方法。

    用法示例:
        # 方式 1（解耦）：传入 config.json 路径
        config = SessionConfig.from_json_file('/data/config.json', 'A01', 'T')

        # 方式 2（兼容）：从项目目录结构加载
        config = SessionConfig.from_dataset('BCICIV_2a', 'A01', 'T')

        print(config.resample_freq)          # 属性访问
        config['ica_exclude'] = [0, 2, 6]    # 字典修改
        config.save()                        # 保存到 config.json
    """

    def __init__(self, dataset_name: str, subject_id: str, session: str,
                 *args, config_path: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset_name = dataset_name
        self._subject_id = subject_id
        self._session = session
        self._config_path = config_path

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
        if not self._config_path:
            raise RuntimeError(
                "未设置 config 文件路径，无法保存。"
                "请使用 from_json_file() 或 from_dataset() 构造，"
                "或手动设置 _config_path。"
            )
        config_path = self._config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r') as f:
            all_cfg = json.load(f)

        defaults = all_cfg.get('default', {})

        # 计算与 default 的差异
        diff = {}
        for key, value in self.items():
            if value is None:
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
    def from_json_file(cls, json_path: str, subject_id: str,
                       session: str) -> 'SessionConfig':
        """从指定 config.json 路径加载配置（解耦版本，不依赖 config.py）。

        外部系统可直接传入 config.json 路径复用 SessionConfig。
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"配置文件不存在: {json_path}")
        with open(json_path, 'r') as f:
            all_cfg = json.load(f)

        params = all_cfg.get('default', {}).copy()
        overrides = all_cfg.get(subject_id, {}).get(session, {})
        params.update(overrides)

        # 从路径推断 dataset_name（如 .../data/BCICIV_2a/config.json）
        dataset_name = os.path.basename(os.path.dirname(json_path))

        return cls(dataset_name, subject_id, session, params,
                   config_path=json_path)

    @classmethod
    def from_dataset(cls, dataset_name: str, subject_id: str,
                     session: str) -> 'SessionConfig':
        """从项目目录结构加载 config.json（兼容接口，需要 config.py）。

        向后兼容，内部委托给 from_json_file。
        """
        from config import get_dataset_dir
        config_path = os.path.join(get_dataset_dir(dataset_name), 'config.json')
        return cls.from_json_file(config_path, subject_id, session)