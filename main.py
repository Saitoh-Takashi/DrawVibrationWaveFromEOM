from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass()
class EOMParams:
    """運動方程式の不変パラメータ（m, k, c, ε）のクラス"""
    m: float  # 質量　[kg]
    k: float  # バネ定数 [N/m]
    c: float  # 粘性係数 [Ns/m]
    epsilon: float  # 偏心質量と質量の重心の距離 [m]


def calc_displacement(time: np.array, static_params: EOMParams, mu: np.array, omega: np.array) -> np.array:
    """変位の時間推移を算出する関数

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (EOMParams): 不変パラメータ（m, k, c, ε）
        mu (np.array): 偏心質量の時間推移
        omega (np.array): 外力の振動数の時間推移

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

    # x=ae^jωtの実部を返す
    return amplitude * np.cos(omega * time)


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


def calc_omega_decrease(time: np.array, static_params: EOMParams, amplification_factor: float) -> np.array:
    """外力の周波数が徐々に減少する場合の時間推移を算出する

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (EOMParams): 不変パラメータ（m, k, c, ε）
        amplification_factor: 外力の周波数の最大値/ωnの値

    Returns:
        np.array: 外力の周波数が徐々に減少する場合の時間推移の配列
    """
    omega_n = np.sqrt(static_params.k / static_params.m)
    return amplification_factor * omega_n * (1 - time / np.max(time))


def calc_omega_increase(time: np.array, static_params: EOMParams, amplification_factor: float) -> np.array:
    """外力の周波数が徐々に増加する場合の時間推移を算出する

    Args:
        time (np.array): 時間（横軸）の配列
        static_params (EOMParams): 不変パラメータ（m, k, c, ε）
        amplification_factor: 外力の周波数の最大値/ωnの値

    Returns:
        np.array: 外力の周波数が徐々に増加する場合の時間推移の配列
    """
    omega_n = np.sqrt(static_params.k / static_params.m)
    return amplification_factor * omega_n * time / np.max(time)


def main():
    # 横軸（時間）の配列を作成
    n = 1000  # データ点数 [-]
    dt = 0.001  # サンプリング周期 [ms]
    t = np.arange(0, n * dt, dt)

    # 不変パラメータ（m, k, c, ε）の設定
    eom_params = EOMParams(m=0.5, k=1000, c=5, epsilon=0.1)

    # 偏心質量の計算
    mu = calc_mu(time=t, amplitude=0.2, decrease_rate=0.5)
    plt.plot(t, mu)
    plt.show()

    # 外力の周波数の計算
    omega = calc_omega_decrease(time=t, static_params=eom_params, amplification_factor=2)
    plt.plot(t, omega)
    plt.show()

    # 変位の算出
    x = calc_displacement(time=t, static_params=eom_params, mu=mu, omega=omega)
    plt.plot(t, x)
    plt.show()


if __name__ == '__main__':
    main()
