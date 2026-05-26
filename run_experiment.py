#!/usr/bin/env python3
"""
Qwen3.5-0.8B 能力修剪实验入口脚本

支持设备: CUDA, ROCm (AMD), MPS (Apple Silicon), CPU
输出: 结构化 JSON/CSV 结果，用于论文撰写和可视化

使用方法:
    # NVIDIA GPU (CUDA)
    python run_experiment.py --device cuda --strategy hybrid --sparsity 0.5

    # AMD GPU (ROCm)
    python run_experiment.py --device rocm --strategy hybrid --sparsity 0.5

    # Apple Silicon (MPS)
    python run_experiment.py --device mps --strategy hybrid --sparsity 0.5

    # 快速测试 (CPU)
    python run_experiment.py --device cpu --strategy layerdrop --sparsity 0.3
"""
import sys
import os
import torch
import argparse
from pathlib import Path
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from capability_pruning import ExperimentConfig, ExperimentRunner


def setup_device(device: str) -> str:
    """自动检测并设置设备"""
    if device == "auto":
        if torch.cuda.is_available():
            # 检查是否是 ROCm 环境
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                print("✅ 检测到 ROCm 环境，自动使用 CUDA (HIP) 接口")
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif device == "rocm":
        print("✅ 指定使用 ROCm，映射到 CUDA (HIP) 接口...")
        return "cuda"
    elif device == "vulkan":
        print("⚠️ 警告: PyTorch 原生不支持 Vulkan 后端进行剪枝训练。")
        print("   建议方案: 在本机完成剪枝并保存模型，然后在 Vulkan 设备上使用 llama.cpp 进行推理测试。")
        print("   当前回退到 CPU 模式...")
        return "cpu"
    return device


def load_model_and_tokenizer(model_path: str, device: str):
    """加载模型和 Tokenizer"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"加载模型: {model_path}")
    print(f"设备: {device}")
    
    # 设置精度
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype if device != "cpu" else "auto",
        device_map=device if device != "cpu" else "cpu",
        local_files_only=True,
    )
    if device == "cpu":
        model = model.float()
    model.eval()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-0.8B 能力修剪实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 混合策略 (推荐)
  python run_experiment.py --strategy hybrid --sparsity 0.5 --device auto

  # Wanda 剪枝
  python run_experiment.py --strategy wanda --sparsity 0.3 --device cuda

  # 仅保留中英文
  python run_experiment.py --languages zh en --device mps
        """
    )
    
    # 实验配置
    parser.add_argument("--model_path", type=str, default="models/qwen/Qwen3___5-0___8B",
                        help="模型路径 (默认: models/qwen/Qwen3___5-0___8B)")
    parser.add_argument("--strategy", type=str, default="hybrid", 
                        choices=["wanda", "layerdrop", "magnitude", "hybrid"],
                        help="修剪策略 (默认: hybrid)")
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="目标稀疏度 0.0-1.0 (默认: 0.5)")
    parser.add_argument("--languages", type=str, nargs="+", default=["zh", "en"],
                        help="保留的语种 (默认: zh en)")
    parser.add_argument("--domains", type=str, nargs="+", default=["stem", "logic"],
                        help="保留的领域 (默认: stem logic)")
    parser.add_argument("--scenarios", type=str, nargs="+", default=["function_calling", "math"],
                        help="保留的场景 (默认: function_calling math)")
    parser.add_argument("--output_dir", type=str, default="results/experiments",
                        help="输出目录 (默认: results/experiments)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "rocm", "mps", "cpu"],
                        help="运行设备 (默认: auto)")
    parser.add_argument("--save_model", action="store_true",
                        help="保存修剪后的模型")
    
    args = parser.parse_args()
    
    # 自动检测设备
    device = setup_device(args.device)
    print(f"使用设备: {device}")
    
    # 解析模型路径
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path
    
    if not model_path.exists():
        print(f"错误: 模型路径不存在: {model_path}")
        print("请确保模型已下载，或指定正确的路径")
        return
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(str(model_path), device)
    
    # 创建配置
    config = ExperimentConfig(
        base_model_path=str(model_path),
        device=device,
        strategy=args.strategy,
        target_sparsity=args.sparsity,
        preserve_languages=args.languages,
        preserve_domains=args.domains,
        preserve_scenarios=args.scenarios,
        output_dir=args.output_dir,
        save_pruned_model=args.save_model,
    )
    
    # 运行实验
    runner = ExperimentRunner(model, tokenizer, config)
    results = runner.run_full_experiment()
    
    # 清理内存
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    
    print(f"\n✅ 实验完成!")
    print(f"📁 结果已保存至: {runner.output_dir}")
    print(f"📊 查看结果:")
    print(f"   - experiment_results.json (完整数据)")
    print(f"   - layer_importance.csv (层重要性矩阵)")
    print(f"   - evaluation_results.csv (评估结果)")
    print(f"   - model_statistics.csv (模型统计)")
    print(f"   - experiment_metadata.csv (实验元数据)")


if __name__ == "__main__":
    main()
