#!/usr/bin/env python3
"""
迁移辅助脚本

功能:
1. 扫描 mmengine 使用情况
2. 生成替换建议
3. 验证替换是否完整
4. 生成迁移报告
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


class MigrationScanner:
    """迁移扫描器"""

    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)
        self.results = defaultdict(list)

    def scan_all(self):
        """扫描所有 Python 文件"""
        print("🔍 扫描 mmengine 使用情况...\n")

        for py_file in self.root_dir.rglob('*.py'):
            if 'tests' in py_file.parts or '__pycache__' in str(py_file):
                continue

            self.scan_file(py_file)

        self.generate_report()

    def scan_file(self, file_path: Path):
        """扫描单个文件"""
        try:
            content = file_path.read_text(encoding='utf-8')

            # 查找 mmengine 导入
            imports = re.findall(r'from mmengine[.\w]* import (.+)', content)
            for imp in imports:
                self.results['imports'].append((str(file_path), imp.strip()))

            # 查找 BaseModule 使用
            if 'BaseModule' in content:
                matches = re.findall(r'class (\w+)\(BaseModule\)', content)
                for match in matches:
                    self.results['BaseModule'].append((str(file_path), match))

            # 查找初始化函数
            init_funcs = ['constant_init', 'kaiming_init', 'xavier_init', 'normal_init']
            for func in init_funcs:
                if func in content:
                    count = len(re.findall(rf'\b{func}\(', content))
                    if count > 0:
                        self.results[func].append((str(file_path), count))

            # 查找 MMLogger
            if 'MMLogger' in content:
                self.results['MMLogger'].append(str(file_path))

            # 查找 Config
            if 'Config.fromfile' in content or 'from mmengine.config import Config' in content:
                self.results['Config'].append(str(file_path))

            # 查找 load_checkpoint
            if re.search(r'from mmengine\.runner import.*load_checkpoint', content):
                self.results['load_checkpoint'].append(str(file_path))

        except Exception as e:
            print(f"⚠️  扫描文件出错 {file_path}: {e}")

    def generate_report(self):
        """生成报告"""
        print("=" * 80)
        print("📊 MMEngine 使用情况报告")
        print("=" * 80)

        # 总览
        total_files = len(set(
            file for files in self.results.values()
            for file in (files if isinstance(files, list) else [files])
            if isinstance(file, str) or (isinstance(file, tuple) and file[0])
        ))
        print(f"\n📁 涉及文件数: {total_files}")

        # BaseModule
        if self.results['BaseModule']:
            print(f"\n🔹 BaseModule 使用 ({len(self.results['BaseModule'])} 处):")
            for file, class_name in self.results['BaseModule']:
                print(f"   - {file}: class {class_name}")

        # 初始化函数
        init_funcs = ['constant_init', 'kaiming_init', 'xavier_init', 'normal_init']
        total_init = sum(
            count for func in init_funcs
            for _, count in self.results.get(func, [])
        )
        if total_init > 0:
            print(f"\n🔹 初始化函数使用 ({total_init} 处):")
            for func in init_funcs:
                if self.results[func]:
                    for file, count in self.results[func]:
                        print(f"   - {file}: {func} ({count}次)")

        # load_checkpoint
        if self.results['load_checkpoint']:
            print(f"\n🔹 load_checkpoint 使用 ({len(self.results['load_checkpoint'])} 处):")
            for file in self.results['load_checkpoint']:
                print(f"   - {file}")

        # MMLogger
        if self.results['MMLogger']:
            print(f"\n🔹 MMLogger 使用 ({len(self.results['MMLogger'])} 处):")
            for file in self.results['MMLogger']:
                print(f"   - {file}")

        # Config
        if self.results['Config']:
            print(f"\n🔹 Config 使用 ({len(self.results['Config'])} 处):")
            for file in self.results['Config']:
                print(f"   - {file}")

        # 所有导入
        if self.results['imports']:
            print(f"\n🔹 mmengine 导入 ({len(self.results['imports'])} 处):")
            import_summary = defaultdict(list)
            for file, imp in self.results['imports']:
                import_summary[imp].append(file)

            for imp, files in sorted(import_summary.items()):
                print(f"   - {imp}:")
                for file in files[:3]:  # 只显示前3个
                    print(f"     · {file}")
                if len(files) > 3:
                    print(f"     · ... 还有 {len(files) - 3} 个文件")

        print("\n" + "=" * 80)
        self.generate_recommendations()

    def generate_recommendations(self):
        """生成替换建议"""
        print("\n💡 替换建议:\n")

        if self.results['BaseModule']:
            print("1️⃣  BaseModule → nn.Module")
            print("   执行:")
            print("   sed -i 's/from mmengine.model import BaseModule/import torch.nn as nn/' basicvsrpp/mmagic/*.py")
            print("   sed -i 's/class \\(.*\\)(BaseModule)/class \\1(nn.Module)/' basicvsrpp/mmagic/*.py")
            print()

        if any(self.results.get(func) for func in ['constant_init', 'kaiming_init', 'xavier_init', 'normal_init']):
            print("2️⃣  初始化函数")
            print("   创建: basicvsrpp/mmagic/pytorch_utils.py")
            print("   替换导入:")
            print("   from basicvsrpp.mmagic.pytorch_utils import constant_init, kaiming_init")
            print()

        if self.results['load_checkpoint']:
            print("3️⃣  load_checkpoint")
            print("   使用: basicvsrpp.mmagic.pytorch_utils.load_checkpoint")
            print()

        if self.results['MMLogger']:
            print("4️⃣  MMLogger")
            print("   替换为: logging.getLogger(__name__)")
            print()

        if self.results['Config']:
            print("5️⃣  Config")
            print("   简化为: dict 或 yaml.safe_load()")
            print()

        print("详细指南请参考: QUICK_MIGRATION_GUIDE.md")


class MigrationValidator:
    """迁移验证器"""

    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)

    def validate(self):
        """验证迁移是否完成"""
        print("\n🔍 验证迁移完成度...\n")

        issues = []

        # 检查是否还有 mmengine 导入
        for py_file in self.root_dir.rglob('*.py'):
            if 'tests' in py_file.parts or '__pycache__' in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8')
                if 'from mmengine' in content or 'import mmengine' in content:
                    # 排除注释
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if ('from mmengine' in line or 'import mmengine' in line) and not line.strip().startswith('#'):
                            issues.append(f"{py_file}:{i} - 仍有 mmengine 导入")
            except Exception as e:
                print(f"⚠️  检查文件出错 {py_file}: {e}")

        # 检查 requirements.txt
        req_file = self.root_dir / 'requirements.txt'
        if req_file.exists():
            content = req_file.read_text()
            if 'mmengine' in content and not content.startswith('#'):
                issues.append("requirements.txt - 仍包含 mmengine 依赖")

        # 报告结果
        if issues:
            print("❌ 发现以下问题:\n")
            for issue in issues:
                print(f"   - {issue}")
            print(f"\n总计: {len(issues)} 个问题")
            return False
        else:
            print("✅ 迁移完成！没有发现 mmengine 依赖")
            return True


def check_pytorch_utils_exists():
    """检查 pytorch_utils.py 是否存在"""
    utils_file = Path('basicvsrpp/mmagic/pytorch_utils.py')
    if not utils_file.exists():
        print("⚠️  pytorch_utils.py 不存在!")
        print("\n建议创建该文件，包含以下内容:")
        print("-" * 60)
        print("""
import torch
import torch.nn as nn
import torch.nn.init as init
import logging

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)

def kaiming_init(module, a=0, mode='fan_in', nonlinearity='leaky_relu', bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)

def load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return model.load_state_dict(state_dict, strict=strict)

def get_logger(name=__name__):
    return logging.getLogger(name)
""")
        print("-" * 60)
        return False
    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='BasicVSR++ 迁移辅助工具')
    parser.add_argument('command', choices=['scan', 'validate', 'check'],
                       help='scan: 扫描使用情况 | validate: 验证迁移 | check: 检查工具文件')
    parser.add_argument('--dir', default='.', help='项目根目录')

    args = parser.parse_args()

    if args.command == 'scan':
        scanner = MigrationScanner(args.dir)
        scanner.scan_all()

    elif args.command == 'validate':
        validator = MigrationValidator(args.dir)
        success = validator.validate()
        sys.exit(0 if success else 1)

    elif args.command == 'check':
        exists = check_pytorch_utils_exists()
        sys.exit(0 if exists else 1)


if __name__ == '__main__':
    main()
