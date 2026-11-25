# 第三方来源与依赖

根目录 MIT 许可证覆盖本项目自有的标定、适配、测试及文档代码。它不重新授权第三方 SDK、模型或依赖。

| 组件 | 来源及处理方式 |
|---|---|
| reBotArm SDK | [Seeed-Projects/reBotArm_control_py](https://github.com/Seeed-Projects/reBotArm_control_py)。通过 `tools/bootstrap_sdks.py` 获取指定提交，版本见 `vendor/sdk-manifest.json`；完整 SDK 与上游网格不放入本仓库。安装与使用遵守上游条款。 |
| SDK 兼容改动 | `vendor/sdk-patches/` 保存本地相对指定上游提交的差异，`vendor/sdk-overlays/` 保存附加的反馈/轨迹等模块。它们属于已测试部署所需的本地扩展，不能把修改后的行为归为未经修改的官方默认行为。补丁涉及的上游代码继续遵循上游条款。 |
| MotorBridge | [tianrking/motorbridge](https://github.com/tianrking/motorbridge)，MIT。保留 `rebot_eye_to_hand/runtime/LICENSE.motorbridge`；提交、逐文件 SHA256 和构建说明见 [依赖说明](docs/INSTALLATION.md)。 |
| Pinocchio / OpenCV / SciPy / NumPy / PyYAML / librealsense | 通过对应项目/包索引安装，遵守各自许可证；不替换各项目的声明。 |
| 演示图片 | 项目作者提供的真实实验截图；仅用于说明界面和操作，并非精度认证。 |

SDK 源文件和 URDF 的版本是标定参数的一部分。更新 SDK 或模型后，应重新核验哈希、角修正配套关系和独立验证，不要只替换模型文件继续沿用旧参数。
