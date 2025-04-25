import torch

print("✅ PyTorch 检查开始...\n")

print(f"🧠 当前 PyTorch 版本: {torch.__version__}")
print(f"🔍 是否支持 CUDA（GPU）: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🚀 可用 GPU 数量: {torch.cuda.device_count()}")
    print(f"💻 当前 GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"📊 当前 CUDA 版本: {torch.version.cuda}")
    print(f"📦 当前设备: {torch.cuda.current_device()} (Index)")
else:
    print("❌ 当前未检测到可用的 CUDA GPU。请检查驱动 / PyTorch 安装是否正确。")
