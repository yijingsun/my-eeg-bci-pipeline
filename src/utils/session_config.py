"""
会话级配置管理器
从 config.json 加载 default + 被试/会话覆盖，提供属性访问、更新、自动 diff 保存
"""
import json
import os
from config import get_dataset_dir


class SessionConfig:
    def __init__(self, dataset_name: str, subject_id: str, session: str):
        self.dataset_name = dataset_name
        self.subject_id = subject_id
        self.session = session
        self._base_dir = get_dataset_dir(dataset_name)
        self._params = {}
        self._load()

    # ---------- 加载 ----------
    def _load(self):
        path = os.path.join(self._base_dir, 'dataset_config.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, 'r') as f:
            self._all_cfg = json.load(f)

        # 1. default
        defaults = self._all_cfg.get('default', {})
        self._params = defaults.copy()

        # 2. 被试/会话覆盖（直接 update）
        sess_params = self._all_cfg.get(self.subject_id, {}).get(self.session, {})
        self._params.update(sess_params)

    # ---------- 通用参数属性 ----------
    @property
    def eog_channels(self) -> list:       return self._params['eog_channels']
    @property
    def resample_freq(self) -> int:       return self._params['resample_freq']
    @property
    def filter_ica(self) -> tuple:        return tuple(self._params['filter_ica'])
    @property
    def filter_design_ica(self) -> str:    return self._params.get('filter_design_ica', 'firwin')
    @property
    def filter_mi(self) -> tuple:         return tuple(self._params['filter_mi'])
    @property
    def filter_design_mi(self) -> str:    return self._params.get('filter_design_mi', 'firwin')
    @property
    def ica_n_components(self) -> int:    return self._params['ica_n_components']
    @property
    def ica_random_state(self) -> int:    return self._params['ica_random_state']
    @property
    def ica_method(self) -> str:          return self._params['ica_method']
    @property
    def ref_type(self) -> str:            return self._params['ref_type']
    @property
    def tmin(self) -> float:              return self._params['tmin']
    @property
    def tmax(self) -> float:              return self._params['tmax']
    @property
    def expected_trials(self) -> int:     return self._params['expected_trials']
    @property
    def mi_event_mapping(self) -> dict:      return self._params['mi_event_mapping']

    # ---------- 手动标记属性 ----------
    @property
    def bad_channels_manual(self) -> list:       return self._params.get('bad_channels_manual', [])
    @property
    def ica_exclude_manual(self) -> list:        return self._params.get('ica_exclude_manual', [])
    @property
    def bad_trials_manual(self) -> list:         return self._params.get('bad_trials_manual', [])

    # ---------- 更新与保存 ----------
    def update(self, key: str, value):
        """更新某个参数值（仅内存）"""
        self._params[key] = value

    def save(self):
        """
        保存当前配置：计算与 default 的差异，写入 config.json 中对应被试/会话下，
        不修改 default 部分。
        """
        path = os.path.join(self._base_dir, 'config.json')
        # 重新读取避免覆盖他人修改
        with open(path, 'r') as f:
            current_cfg = json.load(f)

        defaults = current_cfg.get('default', {})
        diff = {}
        for k, v in self._params.items():
            if k in defaults:
                if v != defaults[k]:
                    diff[k] = v
            else:
                diff[k] = v

        # 存入 subjects.id.session
        current_cfg.setdefault(self.subject_id, {})[self.session] = diff

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(current_cfg, f, indent=2)
        print(f"✓ 配置已保存到 {path}")

    def __repr__(self):
        return (f"SessionConfig(dataset={self.dataset_name}, "
                f"subject={self.subject_id}, session={self.session})")