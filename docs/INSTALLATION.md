# 安装与依赖复现

## 支持范围

已在Linux x86_64、Python3.10、reBot B601-RS、RealSense D435上运行。MotorBridge预构建快照依赖来自Ubuntu24.04，要求glibc≥2.34；其他架构/平台需重新构建，不宣称Windows/macOS或ARM可直接实机运行。

本地实机原有`rebotarm`环境能使用时，无须为了运行本仓库重装它。新机器建议使用独立环境：

```bash
conda env create -f environment.yml
conda activate rebot-calibration
```

Pinocchio使用conda-forge的`pinocchio=3.9`；不要把PyPI的同名非机器人包`pinocchio`装进来。`requirements.txt`中的其余包由pip安装。OpenCV只装一种带ArUco模块的4.10版本，桌面演示需要GUI支持；CI使用独立环境运行离线检查。

Ubuntu桌面常用系统依赖：

```bash
sudo apt-get update
sudo apt-get install git can-utils libgl1 libglib2.0-0
```

RealSense设备权限与USB驱动按[Intel librealsense安装说明](https://github.com/IntelRealSense/librealsense)配置。RealSense接USB3，多个高带宽设备共用Hub时注意带宽与供电；本项目不会自动更换摄像头。

## 固定版本官方SDK

```bash
python tools/bootstrap_sdks.py
```

安装到忽略提交的`vendor/sdks/modern`与`vendor/sdks/legacy`，分别固定在：

| 用途 | 上游提交 |
|---|---|
| 实机演示/自动采集使用的现代SDK基础 | `6415d43130d1e143c70dc106096a857ac5556f81` |
| 手动拖动/腕部原模型使用的旧SDK基础 | `d4f3591bc0183de03f3ae8651298db0269aadfdd` |

现代SDK基础上应用`vendor/sdk-patches/modern.patch`与`vendor/sdk-overlays/modern`。这些是已测试本地部署的兼容扩展，包含严格发送、反馈和轨迹适配；不是重新实现通信库。完整SDK和大型上游网格不加入本仓库。

脚本拒绝覆盖已有目录。已具备官方SDK的本机也可以离线准备：

```bash
python tools/bootstrap_sdks.py \
  --modern-source /path/to/modern-sdk-git-checkout \
  --legacy-source /path/to/legacy-sdk-git-checkout
```

它克隆指定Git对象，再应用本仓库记录的改动，绝不修改源SDK目录。自有SDK不含所需API时，应使用固定版本及补丁，而不是绕过检查。

默认配置已使用仓库相对路径。若使用其他目录：

```bash
python tools/configure_local.py \
  --modern-sdk /path/to/prepared-modern-sdk \
  --legacy-sdk /path/to/prepared-legacy-sdk
```

这一步只改本地配置，不启动相机和机械臂。设置相机序列号还可加`--wrist-serial`与`--external-serial`；已有外参与内参的身份不会被伪改成新相机，新相机应重新采集标定。

## MotorBridge快照运行时

`rebot_eye_to_hand/runtime/`内为未改功能代码的上游快照：

- 上游：https://github.com/tianrking/motorbridge
- 提交：`d5700bcd5b7f9355dd2012488d463c3bac982259`
- 版本：`0.5.6+snapshot.d5700bc`
- MIT许可证：`runtime/LICENSE.motorbridge`
- 文件清单与逐文件校验：`runtime/motorbridge_contract.json`

`eyehand.native_runtime.activate()`在导入现代SDK前检查文件哈希，再为当前Python进程选择这一依赖。不会覆盖环境全局安装。仅安装PyPI的`motorbridge==0.5.6`不能保证具备快照时间戳/序列号API。

不同平台重建：获取上述提交，安装Rust/C++工具链，然后在源码目录执行：

```bash
cargo build --locked --release -p motor_abi -p ws_gateway
cargo test --locked -p motor_abi state_ffi::tests
cargo test --locked -p motor_vendor_robstride
cd bindings/python
python -m pip install setuptools wheel
python setup.py bdist_wheel
```

用同一构建产生的Python包和动态库替换`runtime/python/motorbridge`，保留许可证，重新生成逐文件清单；编译器等变化可能使二进制哈希不同。不能仅替换`.so`而保留不匹配的Python ABI。

## 相机与CAN

```bash
python -c "import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])"
ip -details link show can0
```

作者实验序列号：腕部`332322072170`，外置`250122070127`；RGB1280×720、30Hz。设备身份与分辨率不同，不能继续复用旧内参。

SocketCAN接口须已由本机驱动准备。B601-RS示例使用1Mbps；修改前先核对电机配置，不要在有其他控制进程时改总线：

```bash
sudo ip link set can0 up type can bitrate 1000000
```

## 离线验证

```bash
python tools/verify_release.py
python tools/run_offline_checks.py
```

验证脚本仅做文件、坐标模型、合成几何和模拟控制检查，不连接相机或CAN。原始图片/历史运行日志未发布，对它们有依赖的本地回归用例在公开检查中明确排除，见`tools/run_offline_checks.py`。
