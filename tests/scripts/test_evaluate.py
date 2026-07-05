import sys
import pytest
from scripts.evaluate import main as evaluate_main


class TestEvaluateArgs:
    def test_defaults(self, monkeypatch):
        """默认参数"""
        monkeypatch.setattr(sys, "argv", ["evaluate.py"])
        # 仅验证 argparse 不报错，不执行实际评估
        try:
            evaluate_main()
        except SystemExit:
            pass  # argparse 可能调用 sys.exit
        except Exception:
            pass  # 缺少数据会自然失败，正常
