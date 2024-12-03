import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import matplotlib.style as mplstyle
import tkinter as tk
from PIL import ImageTk, Image
from FFT import fft_solve
from solve import lbm_solver
import time


mplstyle.use('fast')

class initial:
    def __init__(self, lbm):
        self.name = "基于LBM的涡街流量计仿真"
        self.nx = lbm.nx  # 画布尺寸 #为了方便dx、dy、dt均为1
        self.ny = lbm.ny
        self.root = tk.Tk()
        self.root.title(self.name)
        self.root.geometry("{}x{}".format(int(1.69*self.nx),  int(3.1* self.ny)))
        self.font = ("Simhei", 12)
        self.t = 0
        self.f = 0
        colors = [
            (0.7734, 0.07422, 0.01953),  # 红色
            (0.9766, 0.9766, 0.4882),  # 黄色
            (0.1172, 0.1094, 0.5078),  # 蓝色
            (0.9766, 0.9766, 0.4882),  # 黄色
            (0.7734, 0.07422, 0.01953),  # 红色
        ]
        self.my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", colors)

        half_colors = [
            (0.1172, 0.1094, 0.5078),  # 蓝色
            (0.9766, 0.9766, 0.4882),  # 黄色
            (0.7734, 0.07422, 0.01953),  # 红色
        ]
        self.half_my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", half_colors)

        self.Continued_=True #判断动画是否暂停

        self.show_sensor=0

    def start(self,lbm):
        self.v1 = self.v.get()
        self.size_obstacle = self.SizeObstacle.get()#阻流体大小值获取
        self.v_fluid=self.V.get() #流体速度大小获取
        self.rho_fluid=self.rho.get() #黏度大小获取
        lbm.init(self.v1,self.size_obstacle,self.v_fluid,self.rho_fluid)


        self.judge = 1 if self.pic_val.get()==3 else 0
        self.is_firsttime=1
        self.pic_val_type=self.pic_val.get()
        self.last_pic_val=self.pic_val.get()
        self.run(lbm)


    def run(self,lbm):
        global canvas_widget
        self.obsX = []
        self.obsY = []
        self.t=0

        x = np.linspace(0, self.nx, self.nx)
        y = np.linspace(0, self.ny, self.ny)
        X, Y = np.meshgrid(x, y)
        temp = self.size_obstacle * 0.2/ self.v_fluid

        a_new=a_old=0

        while True:  # 当窗口没有关闭时
            if self.Continued_:
                if self.v.get()!=self.v1 or self.SizeObstacle.get()!=self.size_obstacle or self.V.get()!=self.v_fluid or self.rho.get()!=self.rho_fluid:
                    self.start(lbm)

                for i in range(100):
                    lbm.collide_stream()
                    lbm.update_para()
                    lbm.apply_boundary_condition()


                lbm.show_sensor[None]=self.show_sensor
                # print(self.show_sensor)

                vel = lbm.vel.to_numpy()  # 转化为np数组对象
                ugrad = np.gradient(vel[:, :, 0])  # 求x方向速度梯度
                vgrad = np.gradient(vel[:, :, 1])  # 求y方向速度梯度
                vor = ugrad[1] - vgrad[0]  # 计算z方向旋度


                if self.pic_val_type!=self.pic_val.get():
                    self.is_firsttime=1
                    self.pic_val_type=self.pic_val.get()


                if self.pic_val_type==1:
                    if self.judge==1:
                        # 解绑Figure对象和Canvas
                        canvas_widget.get_tk_widget().destroy()
                        self.judge=0
                    # 利用旋度值到RGBA颜色空间
                    vor_img1 = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-self.v_fluid/2800, vmax=self.v_fluid/2800),
                                                 cmap=self.my_cmap).to_rgba(vor)

                    lbm.vor_img.from_numpy(vor_img1)
                    lbm.set_obstacles_color_1()  # 将阻流体的颜色设成白色
                    vor_img = lbm.vor_img.to_numpy()

                    # 将NumPy数组转换为PIL图片对象
                    pic_pil = Image.fromarray((vor_img.transpose(1, 0, 2) * 255).astype(np.uint8))
                    # 将PIL图片对象转换为Tkinter图片对象
                    pic_tk = ImageTk.PhotoImage(pic_pil)
                    self.canvas.delete("all")
                    self.canvas.create_image(self.nx / 2, self.ny / 2, image=pic_tk)

                elif self.pic_val_type==2:
                    if self.judge==1:
                        # 解绑Figure对象和Canvas
                        canvas_widget.get_tk_widget().destroy()
                        self.judge = 0

                    # 利用速度值生成分布图
                    lbm.compute_vel_norm()
                    vel_norm = lbm.vel_norm.to_numpy()
                    # vel_img1 = cm.plasma(vel_norm*20 )


                    # 利用旋度值到RGBA颜色空间
                    vel_img1 = cm.ScalarMappable(
                        norm=matplotlib.colors.Normalize(vmin=-self.v_fluid/1200, vmax=self.v_fluid/1200),
                        cmap=cm.inferno).to_rgba(vel_norm)

                    lbm.vel_img.from_numpy(vel_img1)
                    lbm.set_obstacles_color_2()  # 将阻流体的颜色设成灰色
                    vel_img = lbm.vel_img.to_numpy()

                    # 将NumPy数组转换为PIL图片对象
                    pic_pil2 = Image.fromarray((vel_img.transpose(1, 0, 2) * 255).astype(np.uint8))
                    # 将PIL图片对象转换为Tkinter图片对象
                    pic_tk2 = ImageTk.PhotoImage(pic_pil2)
                    self.canvas.delete("all")
                    self.canvas.create_image(self.nx / 2, self.ny / 2, image=pic_tk2)

                else:
                    # 生成流线图（太慢了可能要插值？）
                    self.canvas.delete("all")
                    if self.judge==0:
                        canvas_widget = FigureCanvasTkAgg(self.fig0, master=self.canvas)
                        canvas_widget.draw()
                        canvas_widget.get_tk_widget().pack()
                        self.judge=1
                    # if int(self.t*100)  % 40==0:#每0.0几秒画一次，降低频率，避免每次画流线图卡顿
                    self.ax0.cla()
                    U = vel.transpose(1, 0, 2)[:, :, 0]
                    V = vel.transpose(1, 0, 2)[:, :, 1]
                    mask = lbm.mask.to_numpy().transpose(1, 0)
                    U = np.ma.array(U, mask=mask)  # 组流体部分忽略，减少计算量

                    self.ax0.streamplot(X, Y, U, V, color=vel.transpose(1, 0, 2)[:, :, 0], linewidth=1, density=[0.5, 0.6])
                    self.ax0.imshow(mask, alpha=0.5, cmap='gray', aspect='auto')  # 将阻流体的颜色设成白色
                    self.fig0.subplots_adjust(left=0, bottom=0, right=1, top=1)

                if self.is_firsttime:
                    self.change_colorbar()
                    self.is_firsttime=0


                # 画受力曲线
                self.t = self.t + 0.01# 每30次迭代画一次 则delta t=0.003

                self.obsX.append(self.t)
                lbm.force()
                a = lbm.force1.to_numpy().sum()
                self.obsY.append(a)
                if self.line is None:
                    self.line = self.ax.plot(self.obsX, self.obsY, '-g')[0]
                # y_smoothed = gaussian_filter1d(obsY, sigma=1)
                if len(self.obsY)>1000:
                    self.obsY = self.obsY[-1000:]
                    self.obsX = self.obsX[-1000:]
                self.line.set_xdata(self.obsX)
                self.line.set_ydata(self.obsY)
                if self.t - 5 >= 0:
                    self.ax.set_xlim([self.t - 5, self.t])
                else:
                    self.ax.set_xlim([0, 5])

                a=np.max(self.obsY[-1000:])
                if a <3:
                    a=3
                self.ax.set_ylim([-1.5*a, 1.5*a])

                if self.line2 is None:
                    self.line2 = self.ax2.plot([], [], '-g')[0]

                x, fft_y = fft_solve(100, self.obsY[-300:])#输入的第一项为采样率，100个dt采一次，每秒采1/0.0001/100次
                if fft_y.any():
                    max_index = np.argmax(fft_y)
                    self.f = x[max_index]/4+0.12/temp

                # if self.v.get()==1:
                #     self.f=self.f*0.8

                self.upgreate_data()
                self.line2.set_xdata(x)
                self.line2.set_ydata(fft_y)

                a=np.max(fft_y) if fft_y.any() else 1
                if a<1:
                    a=1
                self.ax2.set_ylim([0, 1.5*a])
                self.ax2.set_xlim([0, 5])

                self.root.update_idletasks()  # 刷新
                self.root.update()
            else:
                self.root.update_idletasks()  # 刷新
                self.root.update()

    def change_colorbar(self):
        if self.pic_val_type==3:
            if self.last_pic_val!=3:
                self.last_pic_val=3
            self.fig_colorbar.clf()
        else:
            if self.last_pic_val==3:
                self.last_pic_val=0
                self.set_colorbar()

            self.norm = plt.Normalize(vmin=0, vmax=1)

            if self.pic_val_type == 1:
                self.colorbar = matplotlib.colorbar.ColorbarBase(self.ax_colorbar, cmap=self.half_my_cmap,
                                                                 norm=self.norm, orientation='horizontal')

                # Set colorbar ticks and labels
                ticks = [0, 0.25, 0.5, 0.75, 1]
                temp=self.v_fluid/200*0.15
                tick_labels = ['0', '{:.2f}'.format(temp/4), '{:.2f}'.format(temp/2), '{:.2f}'.format(temp*3/4), '{:.2f}'.format(temp)]
                self.colorbar.set_ticks(ticks)
                self.colorbar.set_ticklabels(tick_labels,fontsize=10)

                self.colorbar.set_label(r'涡量大小m/s',font={'family': 'SimHei', 'size': 12})

            else:
                self.colorbar = matplotlib.colorbar.ColorbarBase(self.ax_colorbar, cmap=cm.jet,
                                                                 norm=self.norm, orientation='horizontal')

                # Set colorbar ticks and labels
                ticks = [0, 0.25, 0.5, 0.75, 1]
                temp = self.v_fluid/470*5
                tick_labels = ['0', '{:.2f}'.format(temp / 4), '{:.2f}'.format(temp / 2), '{:.2f}'.format(temp * 3 / 4),
                               '{:.2f}'.format(temp)]
                self.colorbar.set_ticks(ticks)
                self.colorbar.set_ticklabels(tick_labels, fontsize=10)

                self.colorbar.set_label(r'速度大小m/s', font={'family': 'SimHei', 'size': 12})


    def set_colorbar(self):
        # 创建色棒
        self.fig_colorbar = plt.figure(figsize=(3.4, 1), dpi=100)
        self.ax_colorbar = self.fig_colorbar.add_axes([0.01, 0.5, 0.9, 0.2])

        # Get Tkinter default background color
        grey_bg_color = self.root.cget('bg')

        # Convert Tkinter color to Matplotlib color
        grey_rgb_color = self.root.winfo_rgb(grey_bg_color)
        grey_rgb_color = tuple([x / 65536 for x in grey_rgb_color])
        grey_hex_color = matplotlib.colors.rgb2hex(grey_rgb_color)

        # Set the background color to Tkinter's default grey
        self.fig_colorbar.patch.set_facecolor(grey_hex_color)
        self.ax_colorbar.patch.set_facecolor(grey_hex_color)

        # self.ax_colorbar.patch.set_facecolor("#f0f0f0")

        self.colorbar_canvas = FigureCanvasTkAgg(self.fig_colorbar, master=self.root)
        self.colorbar_canvas.get_tk_widget().place(x=self.nx - 280, y=210)

    def set_canvas(self):
        self.set_colorbar()


        self.canvas = tk.Canvas(self.root, width=self.nx, height=self.ny, bg='white')
        self.canvas.place(x=30, y=0)

        # 绘制流线图
        self.fig0 = plt.figure(figsize=(self.nx/100, self.ny/100))
        self.ax0 = self.fig0.add_subplot()


        self.fig = plt.figure(figsize=(6, 3))  # 受力曲线图
        self.ax = self.fig.add_subplot(121)
        self.ax.set_xlabel('Time/s')
        self.ax.set_ylabel('F')
        self.ax.set_title('受力曲线', font={'family': 'SimHei', 'size': 12})
        self.line = None
        plt.grid(True)  # 添加网格

        # Get Tkinter default background color
        grey_bg_color = self.root.cget('bg')

        # Convert Tkinter color to Matplotlib color
        grey_rgb_color = self.root.winfo_rgb(grey_bg_color)
        grey_rgb_color = tuple([x / 65536 for x in grey_rgb_color])
        grey_hex_color = matplotlib.colors.rgb2hex(grey_rgb_color)

        # Set the background color to Tkinter's default grey
        self.fig.patch.set_facecolor(grey_hex_color)

        # plt.subplots_adjust(wspace=2, hspace=0)  # 调整子图间距

        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlabel('Frequency/Hz')
        self.ax2.set_ylabel('N')
        self.ax2.set_title('实时傅里叶变换', font={'family': 'SimHei', 'size': 12})
        self.line2 = None
        plt.grid(True)  # 添加网格
        plt.ion()  # 交互模式


        self.fig.tight_layout(pad=0.4, w_pad=2, h_pad=0)

        self.canvas2 = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().place(x=30, y=self.ny + 120)


    def judge_pause(self):
        if self.Continued_:
            self.Continued_ = False
        else:
            self.Continued_ = True

    def set_radiobutton(self):
        self.set_button_C0 = 80

        label_xinzhuang = tk.Label(self.root, text='漩涡发生体形状：', font=self.font)
        set_radiobutton_C=0
        label_xinzhuang.place(x=self.nx+self.set_button_C0,y=set_radiobutton_C+0)

        # 生成三个选项，分别对应阻流体的三种形状，三角形、圆形、T形
        xinzhuang = [
            ('圆形', 1),
            ('T形', 2),
            ('三角形', 3)]
        self.v = tk.IntVar()  # 这里注意
        self.v.set(1)  # 默认是选第一个

        for lang, num in xinzhuang:
            b = tk.Radiobutton(self.root, text=lang, variable=self.v, value=num, font=self.font)
            b.place(x=self.nx+self.set_button_C0,y=set_radiobutton_C+num*20)

        set_radiobutton_C2=220
        label_plot = tk.Label(self.root, text='显示图像:', font=self.font)
        label_plot.place(x=self.nx+self.set_button_C0,y=set_radiobutton_C2+80)
        plot_to_show = [
            ('涡量分布', 1),
            ('速率分布', 2),
            ('流线图', 3)]
        self.pic_val = tk.IntVar()  # 这里注意
        self.pic_val.set(1)  # 默认是选第一个
        for i, j in plot_to_show:
            butt = tk.Radiobutton(self.root, text=i, variable=self.pic_val, value=j, font=self.font)
            butt.place(x=self.nx+self.set_button_C0,y=set_radiobutton_C2+80+j*20)

    def set_button(self):
        set_button_C1=100

        label_para = tk.Label(self.root, text='参数调节:', font=self.font)
        label_para.place(x=self.nx+self.set_button_C0,y=set_button_C1+0)
        self.label7 = tk.Label(self.root, text='流速：', font=self.font)
        self.label7.place(x=self.nx+self.set_button_C0,y=set_button_C1+20)
        self.V = tk.Scale(self.root, from_=1, to=60,orient=tk.HORIZONTAL, length=200,showvalue=False)
        self.V.set(30)
        self.V.place(x=self.nx+self.set_button_C0,y=set_button_C1+40)

        self.label5 = tk.Label(self.root, text='运动黏度：', font=self.font)
        self.label5.place(x=self.nx+self.set_button_C0,y=set_button_C1+80)
        self.rho = tk.Scale(self.root, from_=1, to=60,orient=tk.HORIZONTAL, length=200,showvalue=False)
        self.rho.set(30)
        self.rho.place(x=self.nx+self.set_button_C0,y=set_button_C1+100)

        self.label6 = tk.Label(self.root, text='漩涡发生体特征长度：', font=self.font)
        self.label6.place(x=self.nx+self.set_button_C0,y=set_button_C1+140)
        self.SizeObstacle = tk.Scale(self.root, from_=1, to=60,orient=tk.HORIZONTAL, length=200,showvalue=False)
        self.SizeObstacle.set(30)
        self.SizeObstacle.place(x=self.nx+self.set_button_C0,y=set_button_C1+160)


        set_button_C2 = -120
        self.lavel0=tk.Label(self.root, text='运行结果：', font=self.font)
        self.lavel0.place(x=self.nx+self.set_button_C0,y=set_button_C2+520)
        self.time = tk.Label(self.root, text='Solution Time:0.00s', font=self.font)
        self.time.place(x=self.nx+self.set_button_C0,y=set_button_C2+540)
        self.label1 = tk.Label(self.root, text='涡旋脱落频率：0Hz', font=self.font)
        self.label1.place(x=self.nx+self.set_button_C0,y=set_button_C2+560)
        self.label2 = tk.Label(self.root, text='流量：0', font=self.font)
        self.label2.place(x=self.nx+self.set_button_C0,y=set_button_C2+580)
        self.label3 = tk.Label(self.root, text='雷诺数Re：0', font=self.font)
        self.label3.place(x=self.nx + self.set_button_C0, y=set_button_C2 + 600)
        self.label4 = tk.Label(self.root, text='斯特罗哈常数St：0', font=self.font)
        self.label4.place(x=self.nx+self.set_button_C0,y=set_button_C2+620)

        # 创建暂停和继续按钮
        self.pause_button = tk.Button(self.root, text="暂停/继续", command=lambda:self.judge_pause(), font=self.font)
        self.pause_button.place(x=self.nx+self.set_button_C0+210,y=set_button_C2+660, width=100, height=50)

        # 创建暂停和继续按钮
        self.pause_button = tk.Button(self.root, text="显示/隐藏受力传感器", command=lambda: self.change_show_sensors(), font=self.font)
        self.pause_button.place(x=self.nx + self.set_button_C0, y=set_button_C2 + 660, width=180, height=50)

    def change_show_sensors(self):
        if self.show_sensor==1:
            self.show_sensor=0
        else:
            self.show_sensor=1
    def upgreate_data(self):
        self.time['text'] = '模拟时间:{:.2f}s'.format(self.t)
        self.label1['text'] = '涡旋脱落频率：{:.2f}Hz'.format(self.f)
        self.label2['text'] = r'流量：{:.3f}S (其中S为横截面面积)'.format(self.v_fluid/200)
        self.label3['text'] = '雷诺数Re：{:.3f}'.format(self.v_fluid/200/((self.rho_fluid+5)*0.0002*2.5e-3)*self.size_obstacle*0.0005*2)
        self.label4['text'] = '斯特罗哈常数St：{:.3f}'.format(self.f*self.size_obstacle*0.001/(self.v_fluid/200))

        self.label5['text'] = '运动黏度：{:.2e} m^2/s'.format((self.rho_fluid+5)*0.0002*2.5e-3)
        self.label6['text'] = '漩涡发生体特征长度：{:.3f} m'.format(self.size_obstacle*0.0005*2)
        self.label7['text'] = '流速：{:.3f} m/s'.format(self.v_fluid/200)


    def UI(self, lbm):
        self.set_radiobutton()
        self.set_button()
        self.set_canvas()
        self.start(lbm)

