import numpy as np
import sys
import os
import warnings

#禁止[Taichi]的输出提示
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
with warnings.catch_warnings():
    with HiddenPrints():
        import taichi as ti
        import taichi.math as tm
        ti.init(arch=ti.gpu)  # 使用GPU计算

#LBM 格子单位与实际物理量单位的转化,由于必须得合理的设置系数避免发散，所以部分参数略奇怪
# 黏度转换系数 9e-4(室温下空气黏度大小1.5e-4)
# 时间转化系数 1e-4s
# 1格对应0，0003m 长度转换系数0.0003
# 速度转换系数 3m/s
# 空气密度1.29kg/m0
# 2.820.3
# 空气流经直径D=0.01 m的圆柱体。自由来流速度为0.15 m/s，流动雷诺数为143，算例采用层流计算。Strouhal数约在0.2左右

@ti.data_oriented
class lbm_solver:
    # 初始化参数
    def __init__(self):
        bc_type = [0, 0, 1, 0]  # 右，上，左，下] 边界条件: 0 -> Dirichlet ; 1 -> Neumann
        bc_value = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        self.name = "涡街流量计仿真"
        self.nx = 600  # 画布尺寸 #为了方便dx、dy、dt均为1
        self.ny = 200

        self.niu = ti.field(dtype=ti.f32, shape=())

        self.cs = 1 / np.sqrt(3)  # 格子声速（LBM里面的概念）
        self.tau = ti.field(dtype=ti.f32, shape=()) # 松弛时间,dt设为1
        self.inv_tau = ti.field(dtype=ti.f32, shape=())
        self.rho = ti.field(float, shape=(self.nx, self.ny))  # 定义nx*ny的密度分布函数ρ
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))  # 定义nx*ny的2维向量的速度分布函数v
        self.mask = ti.field(float, shape=(self.nx, self.ny))  # 0代表此处没有圆柱体，1反之

        self.f_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))  # 定义分布函数f
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0  # 定义九维权重向量并赋值
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1],
                                            [1, -1])  # 定义九维方向向量并赋值

        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.force1=ti.field(float, 50)
        self.vor_img=ti.field(float, shape=(self.nx, self.ny,4))
        self.vel_img = ti.field(float, shape=(self.nx, self.ny, 4))
        self.vel_norm = ti.field(float, shape=(self.nx, self.ny))
        self.cy = 1
        self.cy_para = tm.vec2([200, 100])# 位置

        self.show_sensor = ti.field(dtype=ti.i8, shape=())

    @ti.kernel  # 初始化流场
    def init(self,obstacle_type:ti.i8,obstacle_shape:ti.i16,v_fluid:ti.i8,rho_fluid:ti.i8):
        self.vel.fill(0)  # 令速度均为0
        self.rho.fill(1)  # 令ρ均为1
        self.mask.fill(0)


        # print(self.bc_value[0][0])
        self.bc_value[0][0]=v_fluid/1000
        self.niu[None]=(rho_fluid+5)*0.0002  # 运动黏度
        self.tau[None] = (1 / self.cs ** 2) * self.niu[None] + 0.5  # 松弛时间,dt设为1
        self.inv_tau[None] = 1.0 / self.tau[None]

        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.compute_feq(i, j)
            # 可以修改条件以改变障碍物形状

            #三角形
            if obstacle_type ==3:
                if i > self.cy_para[0] and i <= self.cy_para[0]+obstacle_shape/2 and j<self.ny/2+obstacle_shape and j>self.ny/2-obstacle_shape:
                    self.mask[i, j] = 1.0

                if i > self.cy_para[0]+obstacle_shape/2 and i <= self.cy_para[0]+obstacle_shape*2 and -0.5 * (i - self.cy_para[0] -obstacle_shape*2.5) + self.ny/2 > j > 0.5 * (
                        i - self.cy_para[0] - obstacle_shape*2.5) + self.ny/2:
                    self.mask[i, j] = 1.0

            # 圆形
            if obstacle_type ==1:
                if (i - self.cy_para[0]-obstacle_shape) ** 2 + (j - self.cy_para[1]) ** 2 <= obstacle_shape**2:
                    self.mask[i, j] = 1 #此处有圆柱体

            # T形
            if obstacle_type ==2:
                if i > self.cy_para[0] and i< self.cy_para[0]+obstacle_shape*0.6 and j<self.ny/2+obstacle_shape and j>self.ny/2-obstacle_shape:
                    self.mask[i, j] = 1.0
                if i>=self.cy_para[0]+obstacle_shape*0.6 and i<self.cy_para[0]+obstacle_shape*2 and j<self.ny/2+obstacle_shape*0.5 and j>self.ny/2-obstacle_shape*0.5:
                    self.mask[i, j] = 1.0


    @ti.func  # 计算均衡分布函数f_eq
    def compute_feq(self, i, j):
        eu = self.e @ self.vel[i, j]  # 矩阵乘法
        uv = tm.dot(self.vel[i, j], self.vel[i, j])  # 向量点乘
        return self.w * self.rho[i, j] * (
                    1 + 1 / (self.cs ** 2) * eu + 1 / (2 * self.cs ** 4) * eu * eu - 1 / (2 * self.cs ** 2) * uv)

    @ti.kernel
    def collide_stream(self):  # LBM核心方程，碰撞+流动
        self.f_new.fill(0)
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.compute_feq(ip, jp)
                self.f_new[i, j][k] += (1 - self.inv_tau[None]) * self.f_old[ip, jp][k] + feq[k] * self.inv_tau[None]  # 如果波速没取好会使f变成负的，程序出现bug

    @ti.kernel
    def update_para(self):  # 更新宏观变量ρ、u、v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]
            self.vel[i, j] /= self.rho[i, j]+1e-14  #

    @ti.func
    def apply_boundary_condition0(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:
            # Dirichlet 边界条件
            if self.bc_type[dr] == 0:# [0, 0, 1, 0]
                self.vel[ibc, jbc] = self.bc_value[dr]  # 令边界流体速度恒为[0.1,0] # [[0.05, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            # Neumann 边界条件
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]  # 令边界流体速度等于内圈速度

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = self.compute_feq(ibc, jbc) - self.compute_feq(inb, jnb) + self.f_old[inb, jnb]

    @ti.kernel
    def apply_boundary_condition(self):  # 应用边界条件
        # 对左右
        for j in range(1, self.ny - 1):
            self.apply_boundary_condition0(1, 0, 0, j, 1, j)
            self.apply_boundary_condition0(1, 2, self.nx - 1, j, self.nx - 2, j)
        # 对上下
        for i in range(self.nx):
            self.apply_boundary_condition0(1, 1, i, self.ny - 1, i, self.ny - 2)
            self.apply_boundary_condition0(1, 3, i, 0, i, 1)

        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1:
                self.vel[i, j] = 0, 0 #碰撞后速度变为0
                for k in ti.static(range(9)):
                    ip = i - self.e[k, 0]
                    jp = j - self.e[k, 1]
                    for m in ti.static(range(1, 9)):
                        if self.e[k, 0] == -self.e[m, 0]:
                            if self.e[k, 1] == -self.e[m, 1]:
                                self.f_new[ip, jp][m] +=self.f_new[i,j][k]
                                self.f_new[i,j][k]=0

    @ti.kernel
    def force(self): #计算传感器点压强
        self.force1.fill(0)
        a=400
        b=100

        for i in self.force1:
            self.force1[i] = 2 * (self.e[4, 1] * (self.f_new[a, b+2][4] - self.f_new[a+i, b-2][2]) + self.e[7, 1] * (
                    self.f_new[a+i, b+2][7] - self.f_new[a+i, b-2][6]) + self.e[8, 1] * (
                              self.f_new[a+i, b+2][8] - self.f_new[a+i, b-2][5]))

        # return tm.sqrt(tm.dot(self.vel[300,100],self.vel[300,100]))

    #设置障碍物颜色
    @ti.kernel
    def set_obstacles_color_1(self):
        for i,j,k in self.vor_img:
            if self.mask[i,j]==1:
                self.vor_img[i,j,k]=1


            if i < 400 and i > 360 and j < 102 and j > 98:
                if self.show_sensor[None]==1:
                    if k==0:
                        self.vor_img[i, j, k] = 1
                    else:
                        self.vor_img[i, j, k] = 0

    @ti.kernel
    def set_obstacles_color_2(self):
        for i, j, k in self.vel_img:
            if self.mask[i, j] == 1:
                self.vel_img[i, j, k] = 1

            if i < 400 and i > 360 and j < 102 and j > 98:
                if self.show_sensor[None]==1:
                        self.vel_img[i, j, k] = 0.5

    @ti.kernel
    def compute_vel_norm(self):
        for i,j in self.vel_norm:
            self.vel_norm[i,j]=ti.sqrt(self.vel[i,j][0]**2 + self.vel[i,j][1]**2)
            if self.mask[i, j] == 1:
                self.vel_norm[i, j] = 0

