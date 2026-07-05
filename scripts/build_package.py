#!/usr/bin/env python3
"""
打包构建脚本

用法:
    python scripts/build_package.py              # 构建 sdist + wheel
    python scripts/build_package.py --clean      # 仅清理构建产物
    python scripts/build_package.py --verify     # 构建后验证包完整性
    python scripts/build_package.py --upload     # 构建并上传到 PyPI (测试源)
    python scripts/build_package.py --upload-prod # 构建并上传到 PyPI (正式源)
"""
import argparse
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(PROJECT_ROOT, 'dist')
BUILD_DIR = os.path.join(PROJECT_ROOT, 'build')
EGG_INFO_DIR = os.path.join(PROJECT_ROOT, 'my_eeg_bci_pipeline.egg-info')


def run(cmd: list[str], cwd: str = PROJECT_ROOT):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ 命令失败:\n{result.stderr}")
        sys.exit(1)
    if result.stdout:
        print(result.stdout.strip())
    return result


def clean():
    """清理所有构建产物"""
    for d in [DIST_DIR, BUILD_DIR, EGG_INFO_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"✓ 已删除: {d}")
    # 清理 __pycache__
    for root, dirs, _ in os.walk(PROJECT_ROOT):
        if '.myvenv' in root or '.git' in root:
            continue
        for d in dirs:
            if d == '__pycache__':
                path = os.path.join(root, d)
                shutil.rmtree(path)
    print("✓ 已清理所有 __pycache__")


def check_build_deps():
    """检查构建依赖是否已安装"""
    try:
        import build  # noqa: F401
    except ImportError:
        print("✗ 缺少构建依赖，请先安装:")
        print("  pip install build twine")
        sys.exit(1)


def build_package():
    """构建 sdist 和 wheel"""
    check_build_deps()
    run([sys.executable, '-m', 'build', '--sdist', '--wheel'])
    print(f"\n✓ 构建完成，产物位于: {DIST_DIR}/")
    # 列出产物
    if os.path.exists(DIST_DIR):
        for f in os.listdir(DIST_DIR):
            size = os.path.getsize(os.path.join(DIST_DIR, f)) / 1024
            print(f"  {f}  ({size:.1f} KB)")


def verify_package():
    """验证构建的包"""
    if not os.path.exists(DIST_DIR):
        print("✗ dist/ 目录不存在，请先构建")
        sys.exit(1)

    # 检查 twine
    try:
        import twine  # noqa: F401
    except ImportError:
        print("✗ 缺少 twine，请安装: pip install twine")
        sys.exit(1)

    print("\n=== 包完整性检查 ===")
    # twine check
    dist_files = [os.path.join(DIST_DIR, f) for f in os.listdir(DIST_DIR)]
    run([sys.executable, '-m', 'twine', 'check'] + dist_files)

    # 检查 wheel 内容
    wheels = [f for f in os.listdir(DIST_DIR) if f.endswith('.whl')]
    if wheels:
        import zipfile
        whl_path = os.path.join(DIST_DIR, wheels[0])
        with zipfile.ZipFile(whl_path) as zf:
            names = zf.namelist()
            print("\n=== Wheel 包内容 ===")
            # 检查关键模块
            expected = [
                'src/__init__.py',
                'src/data_preparation/',
                'src/feature_extraction/',
                'src/classification/',
                'src/evaluation/',
                'src/pipeline/',
                'src/utils/',
                'config.py',
            ]
            for exp in expected:
                found = any(exp in n for n in names)
                status = "✓" if found else "✗"
                print(f"  {status} {exp}")

    print("\n✓ 验证完成")


def upload(test: bool = True):
    """上传到 PyPI"""
    if not os.path.exists(DIST_DIR) or not os.listdir(DIST_DIR):
        print("✗ 无构建产物，请先构建")
        sys.exit(1)

    repo = '--test-pypi' if test else '--repository', 'pypi'
    dist_files = [os.path.join(DIST_DIR, f) for f in os.listdir(DIST_DIR)]
    cmd = [sys.executable, '-m', 'twine', 'upload'] + list(repo) + dist_files
    print(f"\n{'上传到 TestPyPI' if test else '上传到 PyPI'}...")
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description='EEG-BCI Pipeline 打包构建工具')
    parser.add_argument('--clean', action='store_true', help='仅清理构建产物')
    parser.add_argument('--verify', action='store_true', help='构建后验证包完整性')
    parser.add_argument('--upload', action='store_true', help='上传到 TestPyPI')
    parser.add_argument('--upload-prod', action='store_true', help='上传到 PyPI 正式源')
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.clean:
        clean()
        return

    # 清理旧产物
    clean()

    # 构建
    build_package()

    # 验证
    if args.verify:
        verify_package()

    # 上传
    if args.upload:
        upload(test=True)
    elif args.upload_prod:
        upload(test=False)


if __name__ == '__main__':
    main()

