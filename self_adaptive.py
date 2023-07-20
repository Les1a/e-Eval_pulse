import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft, signal


def bandpass(ori, low, high, fs):
    Wn = [2 * low / fs, 2 * high / fs]
    b, a = signal.butter(4, Wn, btype='band')
    out = signal.filtfilt(b, a, ori)

    return out


def find_best_n(sign, bg):
    n_values = np.arange(0, 10, 0.01)  # 可以根据需要调整n的范围和步长
    mse_values = []

    for n in n_values:
        diff = sign - n * bg
        mse = np.mean(diff ** 2)
        mse_values.append(mse)

    best_n = n_values[np.argmin(mse_values)]
    min_mse = np.min(mse_values)

    return best_n, min_mse


def load_data(data_path='./data/150-130.csv'):
    event_points = pd.read_csv(data_path, names=['t', 'x', 'y', 'p'])  # names=['x', 'y', 'p', 't']

    period_t = 1 / fps * 1e6  # us
    num_frame = event_points.iloc[-1]['t'] // 1e6 * fps
    print(num_frame)

    '''sum events'''
    img_list = []
    # for n in np.arange(num_frame):
    for n in range(5600, 6080):
        print(n)
        chosen_idx = np.where((event_points['t'] >= period_t * n) * (event_points['t'] < period_t * (n + 1)))[0]
        xypt = event_points.iloc[chosen_idx]
        x, y, p = xypt['x'], xypt['y'], xypt['p']
        p = p * 2 - 1

        img = np.zeros((H, W))
        img[y, x] += p
        img_list.append(img)
    img_list = np.array(img_list)

    '''add mask'''
    face_mask = np.zeros((H, W))
    face_mask[210:275, 175:255] = 1  # face
    face_mask[220:290, 380:455] = 1  # face

    bg_mask = np.zeros((H, W))
    bg_mask[:, :80] = 1  # background
    bg_mask[:, 580:] = 1  # background

    bg_s = np.sum(bg_mask)
    face_s = np.sum(face_mask)

    face_avg = np.sum(img_list * face_mask[np.newaxis], (1, 2)) / face_s
    bg_avg = np.sum(img_list * bg_mask[np.newaxis], (1, 2)) / bg_s

    return face_avg, bg_avg


fps = 120
H = 480
W = 640

if __name__ == '__main__':
    face_avg, bg_avg = load_data()

    n, _ = find_best_n(face_avg, bg_avg)

    sa_signal = np.zeros(len(face_avg))
    for i in range(len(sa_signal)):
        sa_signal[i] = face_avg[i] - n * bg_avg[i]

    bp_sa_signal = bandpass(sa_signal, 1, 4, fps)

    face_avg_bp = bandpass(face_avg, 1, 4, fps)
    bg_avg_bp = bandpass(bg_avg, 1, 4, fps)

    n_bp, _ = find_best_n(face_avg_bp, bg_avg_bp)
    sa_signal_bp = np.zeros(len(face_avg))
    for i in range(len(sa_signal_bp)):
        sa_signal_bp[i] = face_avg_bp[i] - n_bp * bg_avg_bp[i]

    plt.subplot(3, 2, 1)
    plt.plot(face_avg)
    plt.title('face')

    plt.subplot(3, 2, 2)
    plt.plot(bg_avg)
    plt.title('background')

    plt.subplot(3, 2, 3)
    plt.plot(bg_avg_bp)
    plt.title('background_bp')

    plt.subplot(3, 2, 4)
    plt.plot(face_avg_bp)
    plt.title('face_bp')

    plt.subplot(3, 2, 5)
    # plt.plot(bg_avg_bp)
    x = bp_sa_signal
    X = np.fft.fft(x)  # 傅里叶变换
    X_mag = np.abs(X)  # 频谱幅度
    freq = np.fft.fftfreq(len(x)) * fps * 60  # 频率(min^-1)
    plt.plot(freq[:len(x) // 2], X_mag[:len(x) // 2], 'b-')
    plt.title('bp_sa_signal')

    plt.subplot(3, 2, 6)
    x = sa_signal_bp
    X = np.fft.fft(x)  # 傅里叶变换
    X_mag = np.abs(X)  # 频谱幅度
    freq = np.fft.fftfreq(len(x)) * fps * 60  # 频率(min^-1)
    plt.plot(freq[:len(x) // 2], X_mag[:len(x) // 2], 'b-')
    plt.title('sa_signal_bp')

    plt.show()
