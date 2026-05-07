import json
import os
from config import get_dataset_dir

def load_session_config(dataset_name: str, subject_id: str, session: str) -> dict:
    """
    从 config.json 读取 default + 被试/会话覆盖，返回合并后的字典
    """
    config_path = os.path.join(get_dataset_dir(dataset_name), 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r') as f:
        all_cfg = json.load(f)

    params = all_cfg.get('default', {}).copy()
    overrides = all_cfg.get(subject_id, {}).get(session, {})
    params.update(overrides)
    return params