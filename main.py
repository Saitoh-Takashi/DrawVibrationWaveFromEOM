import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class StaticParameters:
    """運動方程式の不変パラメータ（m, k, c, ε）のクラス"""
    m: float  # 質量 [kg]
    k: float  # バネ定数 [N/m]
    c: float  # 粘性係数 [Ns/m]
    epsilon: float  # 偏心質量と質量の重心の距離 [m]

    def plot_title(self):
        """プロットのタイトルの文字列を返す

        Returns:
            str: グラフタイトルの文字列
        """
        omega_n = np.sqrt(self.k / self.m)  * 2 * np.pi
        zeta = self.c / (2 * np.sqrt(self.m * self.k))

        return f'm={self.m:.2f} [kg], k={self.k:.2f} [N/m], c={self.c:.2f}[N s/m], ε={self.epsilon:.2f} [m],\nωn={omega_n:.2f} [rad/s], ζ={zeta:.2f} [-]'


def calc_displacement(time: np.array, static_params: StaticParameters, mu: np.array, omega: np.array) -> np.array:
    """変位の時間推移を算出する関数

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (StaticParameters): 不変パラメータ（m, k, c, ε）
        mu (np.array): 偏心質量の時間推移
        omega (np.array): 外力の角振動数の時間推移

    Returns:
        np.array: 変位の時間推移の配列
    """
    # ωn、ζ、δstの計算
    omega_n = np.sqrt(static_params.k / static_params.m) * 2 * np.pi  # 定数
    zeta = static_params.c / (2 * np.sqrt(static_params.m * static_params.k))  # 定数

    delta_st = ((mu * static_params.epsilon) / static_params.m) * ((omega / omega_n) ** 2)  # np.array

    # 振幅の計算
    amplitude = delta_st / np.sqrt(
        ((1 - ((omega / omega_n) ** 2)) ** 2) + ((2 * zeta * (omega / omega_n)) ** 2))  # np.array

    # 位相の計算
    theta = np.arctan2(-2 * zeta * (omega / omega_n), 1 - (omega / omega_n) ** 2)

    # x=|a|e^j(ωt-θ)の実部を返す
    return amplitude * np.cos(omega * time + theta)


def calc_mu(time: np.array, initial_value: float, decrease_rate: float) -> np.array:
    """偏心質量の時間推移を計算する
    amplitudeで初期値を指定し、時間の推移にしたがって、decrease_rateで指定した割合まで減少する

    Args:
        time (np.array): 時間（横軸）の配列
        initial_value (float): 偏心質量の初期値 [kg]
        decrease_rate (float): 偏心質量が時間推移にしたがって減少する割合

    Returns:
        np.array: 偏心質量の時間推移の配列
    """
    return initial_value - decrease_rate * initial_value * time / np.max(time)


def calc_omega_decrease(time: np.array, static_params: StaticParameters, amplification_factor: float) -> np.array:
    """外力の角振動数が徐々に減少する場合の時間推移を算出する

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (StaticParameters): 不変パラメータ（m, k, c, ε）
        amplification_factor: 外力の角振動数の最大値/ωnの値

    Returns:
        np.array: 外力の角振動数が徐々に減少する場合の時間推移の配列
    """
    omega_n = np.sqrt(static_params.k / static_params.m) * 2 * np.pi  # [rad/s]
    return amplification_factor * omega_n * (1 - time / np.max(time))


def calc_omega_increase(time: np.array, static_params: StaticParameters, amplification_factor: float) -> np.array:
    """外力の角振動数が徐々に増加する場合の時間推移を算出する

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (StaticParameters): 不変パラメータ（m, k, c, ε）
        amplification_factor: 外力の角振動数の最大値/ωnの値

    Returns:
        np.array: 外力の角振動数が徐々に増加する場合の時間推移の配列
    """
    omega_n = np.sqrt(static_params.k / static_params.m) * 2 * np.pi  # [rad/s]
    return amplification_factor * omega_n * time / np.max(time)


def calc_omega_ratio(static_params: StaticParameters, omega: np.array) -> np.array:
    """ω/ωnの時間推移の配列を算出
    
    Args:
        static_params (StaticParameters): 不変パラメータ（m, k, c, ε）
        omega (np.array): 外力の角振動数の時間推移の配列

    Returns:
        np.array: ω/ωnの時間推移の配列
    """
    omega_n = np.sqrt(static_params.k / static_params.m)  * 2 * np.pi  # [rad/s]
    return omega / omega_n


def plot(static_params: StaticParameters, time: np.array, x: np.array, mu: np.array, omega: np.array) -> None:
    """グラフをプロットする

    Args:
        static_params (StaticParameters): 不変パラメータ（m, k, c, ε）
        time (np.array): 横軸（時間）の配列
        x (np.array): 変位の配列
        mu (np.array): 偏心質量の時間推移の配列
        omega (np.array): 外力の角振動数の時間推移の配列
    """
    fig = plt.figure()
    fig.suptitle(static_params.plot_title())

    # 変位
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(time, x)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('x [m]')

    # ω 第一軸
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(time, omega, label='ω')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('ω [rad/s]')

    # ω/ωn
    ax3 = ax2.twinx()
    ax3.plot(time, calc_omega_ratio(static_params=static_params, omega=omega), alpha=0)
    ax3.plot(time, np.ones_like(time), label='ω=ωn')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('ω/ωn [-]')

    # 凡例
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc='upper left')

    # 偏心質量
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
    eom_params = StaticParameters(m=30, k=1080, c=55, epsilon=0.2)
    # 外力の角振動数の計算
    omega = calc_omega_decrease(time=t, static_params=eom_params, amplification_factor=2.22)
    # 偏心質量の計算
    mu = calc_mu(time=t, initial_value=6, decrease_rate=0)
    # 変位の算出
    x = calc_displacement(time=t, static_params=eom_params, mu=mu, omega=omega)

    plot(static_params=eom_params, time=t, x=x, omega=omega, mu=mu)


if __name__ == '__main__':
    main()
