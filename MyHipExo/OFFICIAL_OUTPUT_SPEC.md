# 林-Ⅰ V1 官方输出指标核对

核对日期：2026-10-27。来源为用户提供的 [DRVEXO-Tech 官方仓库](https://github.com/DRVEXO-Tech/exoskeleton-embedded-resources)，核对时 main 的提交为 `dadd9d543d9a4102f762fb453c67232aff1ed1a9`。

## 已确认与尚未给出的指标

| 项目 | 官方资料 | 对当前仿真的含义 |
|---|---|---|
| 产品 | 林-Ⅰ髋关节外骨骼机器人，林-Ⅰ V1 | 使用说明书 V1.0，2026/08 |
| 峰值输出 | 20 N·m | 产品表未明确是单侧还是双侧合计；不能当作连续额定扭矩 |
| 连续额定扭矩、峰值持续时间 | 查阅的手册未提供 | 不能从 20 N·m 推出持续输出能力 |
| 最高/额定输出转速、额定机械功率、扭矩—转速曲线 | 查阅的手册未提供 | 尚不能据此建立电机功率/热限制模型 |
| 力矩指令接口 | `User_Motor_Torque_Set(direction, torque)` | 单位和范围以电机协议及固件定义为准 |
| 速度指令接口 | `User_Motor_Velocity_Set(direction, velocity)`，rad/s | 单位明确，不等于提供了允许速度上限 |
| MIT 接口 | 位置、速度、KP、KD、前馈力矩；用户层不作数值限幅 | API 能发送某个数值不代表设备能持续输出该数值 |
| 标称续航 | 满载 ≥3 小时；轻载（40%功率）≥6 小时 | 手册未给出对应扭矩/转速工况，不能用来反推额定机械功率 |

使用说明书的设备参数位于印刷第 4 页（PDF 第 7 页）。二次开发手册见 §3.3.1、§3.3.3 和 §3.4.6。新主板“历程10 三种基本电机模式”的 `APP_Control_SetTorque` / `APP_Control_SetVelocity` 直接将 float 编码到控制消息，没有在这两个接口中给出额定扭矩或速度常数。25/50/75/100 是反馈挡位原始编码，不能当作 N·m。

## 本项目采用的边界

- **质量以用户实测为准。** 用户于本次核对明确要求采用自己的测量结果；现有模型总质量约 2.55 kg，保持现有质量分布和惯量。官方“3.2 kg（含电池）”仅作为资料差异记录，不覆盖实测模型。
- 保留此前用户指定的每侧 ±20 N·m 仿真限值。其数值与官方产品峰值相同，但“每侧、可持续”仍不是手册已确认的性能。没有擅自提高扭矩、补加电池质量或重新缩放惯量。
- 当前是理想刚性绑定和理想扭矩执行器；没有根据未给出的电机特性臆造速度、功率、热衰减或效率曲线。
- 官方 README 说明 main 适用于 2026-10-01 之后的新机器；实际主板/电机固件版本仍需与设备匹配。

## 固定版本来源

- [使用说明书：PDF 第 7 页](https://github.com/DRVEXO-Tech/exoskeleton-embedded-resources/blob/dadd9d543d9a4102f762fb453c67232aff1ed1a9/1.用户手册/林-Ⅰ系列外骨骼使用说明书.pdf#page=7)
- [二次开发手册](https://github.com/DRVEXO-Tech/exoskeleton-embedded-resources/blob/dadd9d543d9a4102f762fb453c67232aff1ed1a9/1.用户手册/二次开发.pdf)
- [新主板电机控制实现](https://github.com/DRVEXO-Tech/exoskeleton-embedded-resources/blob/dadd9d543d9a4102f762fb453c67232aff1ed1a9/2.例程代码/新主板/历程10%20三种基本电机模式/APP/Src/APP_Control.c)
- [机器适用版本说明](https://github.com/DRVEXO-Tech/exoskeleton-embedded-resources/blob/dadd9d543d9a4102f762fb453c67232aff1ed1a9/README.md)
