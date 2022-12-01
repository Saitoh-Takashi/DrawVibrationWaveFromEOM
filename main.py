import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StaticParameters:
    """運動方程式の不変パラメータ（m, k, c, ε）のクラス"""
    n: int  # データ点数
    dt: float  # サンプリング周期 [ms]
    m: float  # 質量 [kg]
    k: float  # バネ定数 [N/m]
    c: float  # 粘性係数 [Ns/m]
    epsilon: float  # 偏心質量と質量の重心の距離 [m]
    mu_initial: float  # 偏心質量の初期値 [kg]
    mu_decrease_rate: float  # 偏心質量の減少率
    max_omega_ratio: float  # ω/ωnの最大値 [-]


@dataclass
class PlotData:
    """グラフ表示用の配列とラベルを保持するクラス"""
    data: np.array  # 配列
    label: str  # ラベル


class CalculatePlotData:
    """不変パラメータおよび時間軸を保持し，各種プロット用データを返すメソッドを保持するクラス

    Attributes:
        static_params (StaticParameters): 不変パラメータ
        time (np.array): 時間（横軸）の配列
        omega_n (float): ωnの値
        zeta (float): ζの値
    """
    def __init__(self, static_params: StaticParameters) -> None:
        """コンストラクタ

        Args:
            static_params (StaticParameters): 不変パラメータ
        """
        self.static_params = static_params
        self.time = np.arange(0, self.static_params.n * self.static_params.dt, self.static_params.dt)
        self.omega_n = np.sqrt(self.static_params.k / self.static_params.m) * 2 * np.pi  # [rad/s]
        self.zeta = self.static_params.c / (2 * np.sqrt(self.static_params.m * self.static_params.k))

    def displacement(self, mu: np.array, omega: np.array) -> PlotData:
        """変位の時間推移を算出する関数

        Args:
            mu (np.array): 偏心質量の時間推移
            omega (np.array): 外力の角振動数の時間推移

        Returns:
            PlotData: 変位の時間推移の配列とラベル
        """
        # δstの計算
        delta_st = ((mu * self.static_params.epsilon) / self.static_params.m) * (
                (omega / self.omega_n) ** 2)  # np.array

        # 振幅の計算
        amplitude = delta_st / np.sqrt(
            ((1 - ((omega / self.omega_n) ** 2)) ** 2) + ((2 * self.zeta * (omega / self.omega_n)) ** 2))  # np.array

        # 位相の計算
        theta = np.arctan2(-2 * self.zeta * (omega / self.omega_n), 1 - (omega / self.omega_n) ** 2)

        # x=|a|e^j(ωt-θ)の実部を返す
        return PlotData(data=amplitude * np.cos(omega * self.time + theta),
                        label=f'c={self.static_params.c} [Ns/m], ζ={self.zeta:.2f} [-]')

    def mu(self) -> PlotData:
        """偏心質量の時間推移を計算する

        Returns:
            PlotData: 偏心質量の時間推移の配列とラベル
        """

        return PlotData(
            data=self.static_params.mu_initial - self.static_params.mu_decrease_rate * self.static_params.mu_initial * self.time / np.max(
                self.time), label=None)

    def omega_decrease(self) -> PlotData:
        """外力の角振動数が徐々に減少する場合の時間推移を算出する

        Returns:
            PlotData: 外力の角振動数が徐々に減少する場合の時間推移の配列とラベル
        """
        return PlotData(data=self.static_params.max_omega_ratio * self.omega_n * (1 - self.time / np.max(self.time)),
                        label='ω')

    def omega_increase(self) -> PlotData:
        """外力の角振動数が徐々に増加する場合の時間推移を算出する

        Returns:
            PlotData: 外力の角振動数が徐々に増加する場合の時間推移の配列とラベル
        """
        return PlotData(data=self.static_params.max_omega_ratio * self.omega_n * self.time / np.max(self.time),
                        label='ω')

    def omega_ratio(self, omega: np.array) -> PlotData:
        """ω/ωnの時間推移の配列を算出

        Args:
            omega (np.array): 外力の角振動数の時間推移の配列

        Returns:
            PlotData: ω/ωnの時間推移の配列とラベル
        """
        return PlotData(data=(omega / self.omega_n), label='ω/ωn')

    def plot_title(self) -> str:
        """グラフタイトルを返す

        Returns:
            str: グラフタイトル
        """
        return f'm={self.static_params.m:.2f} [kg], k={self.static_params.k:.2f} [N/m], ε={self.static_params.epsilon:.2f} [m], ωn={self.omega_n:.2f} [rad/s]'


class Plot:
    """グラフ用のFigureおよびプロットのメソッドを保持する

    Attributes:
        fig (plt.figure): グラフ用Figure
        ax1 (plt.AxesSubplot): x用のサブプロット
        ax2 (plt.AxesSubplot): x用のサブプロット
        ax3 (plt.AxesSubplot): x用のサブプロット
        ax4 (plt.AxesSubplot): x用のサブプロット

    """

    def __init__(self, figsize: Tuple[float] =(8, 8)) -> None:
        """コンストラクタ

        Args:
            figsize (Tuple[float]): グラフのサイズ
        """
        # Figureの作成
        self.fig = plt.figure(figsize=figsize)

        # x
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax1.set_xlabel('Time [s]')
        self.ax1.set_ylabel('x [m]')

        # ω
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax2.set_xlabel('Time [s]')
        self.ax2.set_ylabel('ω [rad/s]')

        # ω/ωn
        self.ax3 = self.ax2.twinx()
        self.ax3.set_xlabel('Time [s]')
        self.ax3.set_ylabel('ω/ωn [-]')

        # mu
        self.ax4 = self.fig.add_subplot(3, 1, 3)
        self.ax4.set_xlabel('Time [s]')
        self.ax4.set_ylabel('mu [kg]')

    def displacement(self, time: np.array, x: PlotData) -> None:
        """変位の配列をプロット

        Args:
            time (np.array): 時間（横軸）の配列
            x (PlotData): 変位の配列とラベル
        """
        self.ax1.plot(time, x.data, label=x.label)

    def omega(self, time: np.array, omega: PlotData, omega_ratio: PlotData) -> None:
        """ωの時間推移の配列をプロット

        Args:
            time (np.array): 時間（横軸）の配列
            omega (PlotData): 外力の角振動数の時間推移の配列とラベル
            omega_ratio (PlotData): ω/ωnの時間推移の配列とラベル
        """
        self.ax2.plot(time, omega.data, label=omega.label)
        self.ax3.plot(time, omega_ratio.data, alpha=0)
        self.ax3.plot(time, np.ones_like(time), label='ω=ωn')

    def mu(self, time: np.array, mu: PlotData) -> None:
        """muの時間推移の配列をプロット

        Args:
            time (np.array): 時間（横軸）の配列
            mu (PlotData): 偏心質量の時間推移の配列
        """
        self.ax4.plot(time, mu.data)

    def save(self, file_name: str) -> None:
        """ω/ωnの時間推移の配列をプロット

        Args:
            file_name (str): 保存名
        """
        # 凡例
        h1, l1 = self.ax2.get_legend_handles_labels()
        h2, l2 = self.ax3.get_legend_handles_labels()
        self.ax1.legend()
        self.ax3.legend(h1 + h2, l1 + l2)

        # レイアウトの調節
        self.ax1.minorticks_on()
        self.fig.tight_layout()

        # グラフの保存，表示
        plt.savefig(f'{file_name}.svg')
        plt.show()


def main():
    # 不変パラメータの設定
    eom_params = [
        StaticParameters(n=10000,
                         dt=0.001,
                         m=35,
                         k=1080,
                         c=c,
                         epsilon=0.2,
                         mu_initial=5,
                         mu_decrease_rate=0,
                         max_omega_ratio=2.22) for c in [30, 50, 100]]

    # グラフの用意
    figure = Plot()

    # 不変パラメータごとに計算，描写
    for index, static_params in enumerate(eom_params):
        # プロット用クラスの呼び出し
        plotdata_calculator = CalculatePlotData(static_params=static_params)

        # 外力の角振動数の計算
        omega = plotdata_calculator.omega_decrease()
        omega_ratio = plotdata_calculator.omega_ratio(omega=omega.data)
        # 偏心質量の計算
        mu = plotdata_calculator.mu()
        # 変位の算出
        x = plotdata_calculator.displacement(mu=mu.data, omega=omega.data)

        # 変位をプロット
        figure.displacement(time=plotdata_calculator.time, x=x)
        # ω，muをプロット
        if index == 0:
            figure.fig.suptitle(plotdata_calculator.plot_title())
            figure.omega(time=plotdata_calculator.time, omega=omega, omega_ratio=omega_ratio)
            figure.mu(time=plotdata_calculator.time, mu=mu)

    # 保存
    figure.save(file_name='Plot')


if __name__ == '__main__':
    main()
