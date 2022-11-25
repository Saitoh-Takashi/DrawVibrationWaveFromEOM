from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class StaticParams:
    """運動方程式の不変パラメータ（m, k, c, ε）のクラス"""
    m: float  # 質量　[kg]
    k: float  # バネ定数 [N/m]
    c: float  # 粘性係数 [Ns/m]
    epsilon: float  # 偏心質量と質量の重心の距離 [m]

    def plot_title(self):
        omega_n = np.sqrt(self.k / self.m)  # 定数
        zeta = self.c / (2 * np.sqrt(self.m * self.k))  # 定数

        return f'm={self.m:.2f}, k={self.k:.2f}, c={self.c:.2f}, ε={self.epsilon:.2f},\nωn={omega_n:.2f}, ζ={zeta:.2f}'


def calc_displacement(time: np.array, static_params: StaticParams, mu: np.array, omega: np.array) -> np.array:
    """変位の時間推移を算出する関数

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (StaticParams): 不変パラメータ（m, k, c, ε）
        mu (np.array): 偏心質量の時間推移
        omega (np.array): 外力の角振動数の時間推移

    Returns:
        np.array: 変位の時間推移の配列
    """
    # ωn、ζ、δstの計算
    omega_n = np.sqrt(static_params.k / static_params.m)  # 定数
    zeta = static_params.c / (2 * np.sqrt(static_params.m * static_params.k))  # 定数
    delta_st = ((mu * static_params.epsilon) / static_params.m) * ((omega / omega_n) ** 2)  # np.array

    # 振幅の計算
    amplitude = delta_st / np.sqrt(
        ((1 - ((omega / omega_n) ** 2)) ** 2) + ((2 * zeta * (omega / omega_n)) ** 2))  # np.array

    # 位相の計算
    theta = np.arctan2(-2 * zeta * (omega / omega_n), 1 - (omega / omega_n) ** 2)

    # x=|a|e^j(ωt-θ)の実部を返す
    return amplitude * np.cos(omega * time - theta)


def calc_frequency_ratio(static_params: StaticParams, omega: np.array) -> np.array:
    omega_n = np.sqrt(static_params.k / static_params.m)  # 定数
    return omega / omega_n


def calc_mu(time: np.array, amplitude: float, decrease_rate: float) -> np.array:
    """偏心質量の時間推移を計算する
    amplitudeで初期値を指定し、時間の推移にしたがって、decrease_rateで指定した割合まで減少する

    Args:
        time (np.array): 時間（横軸）の配列
        amplitude (float): 偏心質量の初期値 [kg]
        decrease_rate (float): 偏心質量が時間推移にしたがって減少する割合

    Returns:
        np.array: 偏心質量の時間推移の配列
    """
    return amplitude - decrease_rate * amplitude * time / np.max(time)


def calc_omega_decrease(time: np.array, static_params: StaticParams, amplification_factor: float) -> np.array:
    """外力の角振動数が徐々に減少する場合の時間推移を算出する

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (StaticParams): 不変パラメータ（m, k, c, ε）
        amplification_factor: 外力の角振動数の最大値/ωnの値

    Returns:
        np.array: 外力の角振動数が徐々に減少する場合の時間推移の配列
    """
    omega_n = np.sqrt(static_params.k / static_params.m)
    return amplification_factor * omega_n * (1 - time / np.max(time))


def calc_omega_increase(time: np.array, static_params: StaticParams, amplification_factor: float) -> np.array:
    """外力の角振動数が徐々に増加する場合の時間推移を算出する

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (StaticParams): 不変パラメータ（m, k, c, ε）
        amplification_factor: 外力の角振動数の最大値/ωnの値

    Returns:
        np.array: 外力の角振動数が徐々に増加する場合の時間推移の配列
    """
    omega_n = np.sqrt(static_params.k / static_params.m)
    return amplification_factor * omega_n * time / np.max(time)


def plot(static_params: StaticParams, time: np.array, x: np.array, mu: np.array, omega: np.array) -> None:
    """グラフをプロットする

    Args:
        static_params (StaticParams): 不変パラメータ（m, k, c, ε）
        time (np.array): 横軸（時間）の配列
        x (np.array): 変位の配列
        mu (np.array): 偏心質量の時間推移の配列
        omega (np.array): 外力の角振動数の時間推移の配列
    """
    fig = plt.figure()
    fig.suptitle(static_params.plot_title())
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(time, x)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('x [m]')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(time, omega)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('ω [rad/s]')

    ax3 = ax2.twinx()
    ax3.plot(time, calc_frequency_ratio(static_params=static_params, omega=omega), alpha=0)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('ω/ωn [-]')

    ax4 = fig.add_subplot(3, 1, 3)
    ax4.plot(time, mu)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('mu [kg]')

    fig.tight_layout()

    plt.show()


def main():
    # 横軸（時間）の配列を作成
    n = 10000  # データ点数 [-]
    dt = 0.001  # サンプリング周期 [ms]
    t = np.arange(0, n * dt, dt)

    # 不変パラメータ（m, k, c, ε）の設定
    eom_params = StaticParams(m=0.5, k=1000, c=5, epsilon=0.1)
    # 偏心質量の計算
    mu = calc_mu(time=t, amplitude=0.2, decrease_rate=0.5)
    # 外力の角振動数の計算
    omega = calc_omega_increase(time=t, static_params=eom_params, amplification_factor=2)
    # 変位の算出
    x = calc_displacement(time=t, static_params=eom_params, mu=mu, omega=omega)

    plot(static_params=eom_params, time=t, x=x, omega=omega, mu=mu)


if __name__ == '__main__':
    main()
