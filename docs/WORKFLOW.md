# 两种手眼标定：实机教学与复现步骤

所有路径相对仓库根目录。先读[安装说明](INSTALLATION.md)。参数以当前配置文件为准，文中的作者示例不代表其他安装的真值。

## 1. 先解释这次要标定什么

- 腕部相机：相机随末端运动，码板固定桌面，求相机相对末端的六自由度外参。
- 外置相机：相机相对基座固定，码板标定时刚性装在末端，求相机相对基座的六自由度外参。
- RGB内参是`fx/fy/cx/cy`与畸变系数，外参是3D旋转和平移。普通RGB相机加已知尺寸靶标即可标定六自由度外参。
- 码板尺寸量黑色外框，不含白边。本默认新采集使用`DICT_4X4_50`、ID0、黑框60mm。作者已保存的74组腕部示例是在100mm码上训练的；角点演示后来改回60mm，这两项不能混为一项。

## 2. 安装状态与负载核对

| 工况 | 码板位置 | 腕部相机 | 两根夹指 | 默认点质量/减重 |
|---|---|---|---|---|
| 腕部自动采集 | 固定裸桌面 | 安装，110g | 已拆 | 相机+110g；左右夹指各减18g |
| 腕部手动验证/指尖演示 | 固定桌面，可按演示架高 | 安装，110g | 装回 | 相机+110g，不扣夹指 |
| 外置自动采集 | 刚性固定末端 | 已移除 | 装回 | 末端码板+16g |
| 外置角点演示 | 固定桌面 | 已移除 | 装回 | 末端码板0g |

这些只是作者配置，不会自动检测实物是否一致。实际未拆夹指，应将腕部自动采集`config/auto_collection.json`里的`gravity_link_mass_reductions_kg`改为`{}`。相机/支架质量和重心变更后，修改对应`gravity*.json`。自动采集中的减重只作用于私有重力模型，不改磁盘URDF、FK或编码器零点。

TCP示例是两根夹指最前端的中点，在模型`gripper_end`原点。夹指拆下时该点是虚拟参考点，不能声称当前实物已有这个指尖位置。

## 3. 腕部：自动采集

固定码板，相机安装全程不变。采集前结束其他控制器；每条实机指令逐条执行，上一轮保持/失能退出完成后再启动下一轮。

```bash
cd rebot_eye_in_hand
python scripts/collect_handeye_eih_auto.py --preview
```

只预览不开CAN。零位看不到码可正常进入首个观察位后再检测，不要求人为先拖过去。

```bash
python scripts/collect_handeye_eih_auto.py --plan-only --output runs/plan_01
python scripts/collect_handeye_eih_auto.py --execute --output runs/train_01
```

默认150训练候选，包含低位/大俯仰；实际通过IK/路径预检和识别保存的数量以报告为准。RGB30Hz不是每秒保存30个机器人姿态：每处先停稳，再保存图像、关节角、内参与静止窗口信息。

`h`/空格暂停保持，`s`继续，`q`/Esc请求回零。全部完成也请求回零，然后继续保持，托稳后终端输入`release`。连续未识别不自动暂停（`max_consecutive_capture_failures=0`），单姿态8秒超时跳过；真正通信/相机故障仍停止。

独立验证另采一轮：

```bash
python scripts/collect_handeye_eih_auto.py --execute --split validation --output runs/validation_01
```

默认92验证候选，含低位大俯仰。训练/验证目标不重复，但仍需要实际成功保存足够的这些姿态；有候选不代表最终数据覆盖到了。

中断补采示例：

```bash
python scripts/collect_handeye_eih_auto.py --execute \
  --resume-from runs/train_01 --output runs/train_01_resume
```

原目录只读，新目录复制已保存组再继续。相机、码板、负载与配置必须保持不变，不可把60mm码与100mm码的会话续接。输出目录必须未使用。

## 4. 腕部：离线拟合

先作名义运动学基线：

```bash
python scripts/collect_handeye_eih_auto.py fit \
  --dataset runs/train_01 --validation-dataset runs/validation_01 \
  --output runs/baseline_01
```

可选外参＋有效关节角修正，使用从作者74组实验整理出的重投影优化流程：

```bash
python scripts/fit_handeye_eih_paired.py \
  --dataset runs/train_01 --validation-dataset runs/validation_01 \
  --output runs/paired_candidate_01
```

输出`hand_eye_candidate.npz`与`report.json`，第1/6轴固定为0，第2～5轴在±5°范围内拟合；不辨识连杆长度/轴线/柔顺性。验证样本不能进入拟合，也不能混用新外参与旧角修正。两个命令均不开硬件，不会覆盖启用结果。

候选通过独立验证和实测后，先备份`config/calibration`，再将候选NPZ复制为`config/calibration/hand_eye.npz`。运行时从NPZ同时读取外参与角修正；`joint_correction.json`、`active_calibration.json`及实验记录应同步候选内容。保持原始数据和候选，不手工改相机序列号或模型哈希来绕过配对检查。

## 5. 外置：自动采集、拟合与验证

回到仓库根目录，再进入外置项目。**标定时码板在末端，不在桌面。** 外置相机、基座、码板相对夹爪的安装全程不变。

```bash
cd ../rebot_eye_to_hand
python scripts/collect_handeye_eth_auto.py --preview
python scripts/collect_handeye_eth_auto.py --plan-only --output runs/plan_01
python scripts/collect_handeye_eth_auto.py --execute --output runs/train_01
```

默认可见采样区域已前移，适合作者斜置相机视角；其他视角应先改`auto_collection_config.json`的`sampling_region`并离线检查。看不到的姿态不能靠堆积同类样本解决。默认连续识别失败暂停策略见配置，不同于腕部默认0。

```bash
python scripts/collect_handeye_eth_auto.py --execute --split validation \
  --config auto_validation_config.json --output runs/validation_01
python -m eyehand calibrate --dataset runs/train_01 --output runs/baseline_01.json
python -m eyehand identify-kinematics --dataset runs/train_01 --output runs/paired_candidate_01
python -m eyehand validate --result runs/paired_candidate_01/result.json \
  --dataset runs/validation_01 --output runs/validation_report_01.json
```

`identify-kinematics`同时生成外参与配套角修正，不能只摘取其中一个矩阵使用。验证不得复用训练样本。通过后，将`aruco_corners_control_config.json`的`result`指向新的结果文件，保留其引用的原始URDF及哈希。

外置补采：

```bash
python scripts/collect_handeye_eth_auto.py --execute \
  --resume-from runs/train_01 --start-pose 64
```

`64`表示可执行位姿序列的第64处，不是第64组样本；要按实际日志选择，也可省略让程序从最后保存处后继续。新目录含继承与追加样本，不能再与原目录合并一次造成重复。

## 6. 可选：手动重力拖动

腕部：

```bash
cd ../rebot_eye_in_hand
python scripts/validate_handeye_eih.py preview
python scripts/validate_handeye_eih.py collect --manual --output runs/manual_validation_01
python scripts/validate_handeye_eih.py report --dataset runs/manual_validation_01
```

这是冻结当前参数的采集/验证入口。Enter或`s`保存，`q`结束；手动重力采集不自动回零，托稳机械臂和目标后输入`release`。改变姿态时覆盖多个位置、深度及独立旋转轴；不能只转单一轴或只微调相邻关节。

外置：

```bash
cd ../rebot_eye_to_hand
python scripts/collect_handeye_eth.py --manual --manual-backend reference \
  --channel can0 --serial 250122070127 --gravity-config gravity_collection.json \
  --output runs/manual_train_01
```

外置仍需码板刚性固定末端。若使用别的相机，把显式序列号改成实际设备，并重采内外参。

## 7. 四角对准展示

完成标定后，把码板固定到桌面或支架上，量黑框60mm；指尖演示先装回两根夹指。码板架高时，程序用PnP估计四角XYZ。相机、基座及腕部相机安装仍须符合当前参数。

```bash
# 腕部，在rebot_eye_in_hand目录
python scripts/aruco_corners_eih.py --execute --clearance-mm 20
# 外置，在rebot_eye_to_hand目录
python scripts/aruco_corners_eth.py --execute --clearance-mm 20
```

20mm只是首次检查用的示例较大间隙，不写入配置。确认姿态/路径和实测间隙后再减小；作者当前配置分别为腕部3mm、外置0.5mm。

按1～4选角点。腕部每次先回观察位；外置每次先回RAW零位，再获得稳定图像并冻结目标。接近时不因目标被夹爪遮挡就不断重新识别改目标。`r`只回位；`h`/空格保持；`q`/Esc正常回零后保持，托稳输入`release`。

不要求末端严格垂直，程序优先尝试更朝下的可达姿态。不能只看图像里指尖与码板角点重合：指尖在角点上方时，斜视投影本来就可能错开，应比较目标悬停位置或做物理测量。

## 8. 怎样判断效果

依次记录四层结果：

1. **视觉重复性**：相机与码板静止，看PnP位置/角度抖动。外置可运行`check_vision_stability.py --serial ... --seconds 10 --output runs/vision_01.json`。重复性好不代表无系统偏差。
2. **独立固定靶标一致性**：不同机器人姿态下，算出的同一码板基座位姿是否一致，查看平移、旋转及重投影P95；低位和大俯仰单独看。
3. **模型到位**：日志FK残差与跟踪情况；它仅说明控制是否收敛到模型目标。
4. **物理准确性**：用量具/垂线测“两指最前端中点”与角点的XY偏差及真实间隙；同一点重复到达、不同角点、不同末端俯仰分别记录。

若FK仅0.3mm而指尖差几毫米，应检查相机安装/内参、TCP、运动学有效修正和训练姿态覆盖。不要通过不断加固定Z补偿掩盖随姿态变化的误差。

## 9. RGB-D点击演示

```bash
# 腕部目录
python scripts/click_hover_eih.py --execute
# 外置目录
python scripts/click_hover_eth.py --execute
```

鼠标像素需由对齐深度反投影为相机3D点，再转基座坐标。桌面高度按保存参考处理，点击抬高物体不代表要把悬停高度也叠加抬高。腕部发布版清除了作者本地桌面参考，应重新选取/测量。

外置点击保留的是**旧版独立配套模型**，位于`examples/legacy_click`，不是新的外置角点结果；不能拿不同配置程序的BASE读数直接比较来认定外参错了。

## 10. 常见问题

| 现象 | 优先检查 |
|---|---|
| Device or resource busy | 关闭其他RealSense进程；避免两个demo抢同一相机 |
| ArUco未识别 | 字典、ID、黑框尺寸、光照、完整入镜和夹爪遮挡 |
| 反馈缺失/过期 | 停止第二个控制器，查看SDK状态快照序列/年龄、CAN与USB日志；不要用缓存时间掩盖 |
| 接近抖动 | 对比`--outer-loop off`、`settle`和`tracking`的同一路径，检查实际循环耗时与限幅；避免同时改变增益/速度/外环 |
| 角点Z偏差 | 分清板面Z、目标Z、模型TCP Z与实测间隙；检查低位大俯仰覆盖、相机安装和TCP |
| 回零未达标 | RAW零位与修正后模型零位不同；正常流程可在完成轨迹后保持，不代表量具检验达标 |
| 输出已存在 | 使用新目录；不要覆盖旧数据或把不同安装的数据混合 |
