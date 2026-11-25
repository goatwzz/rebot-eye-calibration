# 示例腕部配套标定

`hand_eye.npz`保存74组外参与配套有效关节角修正。`joint_correction.json`是同一修正的可读副本，不能修改编码器零点或重复叠加。

当前角点间隙为3mm，来源为上级`aruco_corners.json`；它不是标定参数。

[完整结果与局限](../../../docs/RESULTS.md)，[验证数值](../../../examples/calibration/eye_in_hand/validation_report.json)。路径相对仓库根目录时请从结果文档进入；NPZ中的模型哈希和相机身份必须与当前硬件匹配。

`intrinsics.npz`保留作旧格式读取兼容，运行中的RGB内参使用`config/wrist_rgb_intrinsics.json`。
