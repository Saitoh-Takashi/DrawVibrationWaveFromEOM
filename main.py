import numpy as np
import matplotlib.pyplot as plt


def calc_displacement(time: np.array) -> np.array:
    """変位を算出する関数
    Args:
        time (np.array): 時間（横軸）の配列

    Returns:
        np.array: 変位の配列
    """
    return np.sin(time) * 10


def main():
    N = 1000
    dt = 0.01

    t = np.arange(0, N * dt, dt)

    plt.plot(t, calc_displacement(t))
    plt.show()


if __name__ == '__main__':
    main()
