# Condor Hopper

通过 Python 和 kRPC 控制 KSP 中的载具：默认全油门竖直上升至**海拔 10000 米**后熄火，保持姿态滑行，下降后用栅格舵修正落点，再重新点火减速着陆。

着陆目标为**南纬 0°12′21″、西经 74°28′23″**（`-0.20583333, -74.47305556`），距发射台约 1442 米。程序从游戏读取目标地面海拔，落点允许误差为 **20 米**。

## 准备与运行

需要 Python 3.11 或更新版本，以及已开启 kRPC 的 KSP。当前程序按一台可重新点火的发动机、四片栅格舵和四条支腿检查载具。

试飞前将飞行器放回发射台，恢复燃料，收起栅格舵和支腿，保持静止、竖直。游戏应正常播放；数据停止更新超过三秒，程序会停止控制。

在本目录安装命令和依赖：

```powershell
python -m pip install -e .
```

安装后可在任意目录使用：

```powershell
condor-hopper probe
condor-hopper fly
condor-hopper fly --target-altitude 20000
condor-hopper --help
```

`probe` 只检查状态，输出载具、燃料、目标坐标和距离。`fly` 直接执行一次完整飞行，默认海拔 **10000 米**熄火，无需 `--execute`；`--target-altitude` 可指定其他熄火海拔。直接输入 `condor-hopper` 或使用 `--help` 查看命令说明。

仍可在项目目录使用 `python main.py probe` 和 `python main.py fly`。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--target-altitude` | `10000` | 发动机熄火的海拔，单位米；只用于 `fly` |
| `--address` | `127.0.0.1` | kRPC 服务器地址 |
| `--rpc-port` | `50000` | 控制连接端口 |
| `--stream-port` | `50001` | 飞行数据端口 |

目标坐标保存在 [vehicle.py](condor_hopper/vehicle.py) 的 `target_latitude` 和 `target_longitude` 中。修改后先运行 `probe` 核对。

## 飞行过程

1. **上升**：使用 kRPC 自动驾驶保持竖直，全油门上升，中途不减速。达到熄火海拔后关闭发动机，飞行器仍会继续上升。
2. **滑行**：保持方向，确认已经下降后才展开栅格舵。下降姿态由 Python 控制，保持机头朝上、发动机端先行。
3. **修正落点**：从目标上方 12000 米以下开始计算点火和减速路线，把点火后的转向、空气作用和落点一起考虑。栅格舵朝这条路线的点火位置和速度调整，采纳点火时刻后不再推迟。
4. **点火着陆**：跟随同一条路线，按实测位置、速度和空气作用修正推力；偏差超过 20 米或 5 米/秒时重新计算后续路线。各段都保留发动机独立刹车的余量。发动机喷口和栅格舵共同控制方向，两者侧向作用接近抵消时先消除横移。支腿相对目标地面的高度降至 **800 米**时展开；点火时机仍由程序计算。
5. **接地**：最后 200 米朝反速度方向并稍向上偏转，只减速、消除水平速度，不再追逐位置偏差，也不会因高度低就强行转正。按当前位置的真实离地高度减速，横移消失后自然转正；第一条支腿接地就熄火并停止转向，随后检查是否稳定。

着陆采用在下降途中点火、连续减速至接地的方式，不通过近地面悬停或反复平移追点。规划把落点收进目标周围 10 米，为实际控制留出误差，最终按 20 米检查。数学计算参考公开的凸优化着陆研究；公式、论文链接和模型限制见 [LANDING.md](LANDING.md)。

程序确认成功需要同时满足：

| 检查项 | 条件 |
| --- | --- |
| 落点偏差 | 不超过 20 米 |
| 升降速度 | 绝对值不超过 1 米/秒 |
| 水平速度 | 不超过 0.5 米/秒 |
| 箭体倾角 | 不超过 3° |
| 接地稳定 | 至少三条支腿接地，速度和姿态满足上述条件，持续一秒 |

## 代码位置

- [cli.py](condor_hopper/cli.py)：`condor-hopper` 命令入口、飞行循环和记录。
- [main.py](main.py)：保留直接运行 Python 脚本的入口。
- [vehicle.py](condor_hopper/vehicle.py)：读取游戏数据、检查载具、控制发动机与舵面。
- [mission.py](condor_hopper/mission.py)：飞行阶段、姿态控制与末段减速。
- [landing.py](condor_hopper/landing.py)：点火时机、轨迹计算和运动检查。
- [CHANGELOG.md](CHANGELOG.md)：改动记录。

kRPC 0.5.2 的舵面转动力带有正负号；Python 按各部件转动力的幅值计算，避免相加时抵消。姿态控制按当前读出的转向能力计算操纵量，避免低估栅格舵作用后输入过大、来回摆动。上升和滑行仍使用原有自动驾驶，下降不依赖服务端自动驾驶的转向计算。

kRPC 参考：[飞行数据](https://krpc.github.io/krpc/latest/python/api/space-center/flight.html)、[自动驾驶](https://krpc.github.io/krpc/latest/python/api/space-center/auto-pilot.html)、[部件](https://krpc.github.io/krpc/latest/python/api/space-center/parts.html)。
