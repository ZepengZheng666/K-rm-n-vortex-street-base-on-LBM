import numpy as np
from scipy.fftpack import fft

def fft_solve(Fs,y):
    fft_y = fft(y)  # 使用快速傅里叶变换，得到的fft_y是长度为N的复数数组
    N=len(y)
    x = np.arange(N)*Fs/N  # 频率个数
    half_x = x[range(int(N / 2))]  # 取一半区间
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模
    normalization_y = abs_y / (N / 2)  # 归一化处理（双边频谱）
    normalization_y[0] /= 2
    normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

    return half_x,normalization_half_y
