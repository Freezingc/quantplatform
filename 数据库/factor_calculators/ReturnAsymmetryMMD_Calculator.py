from .base import FactorCalculator
import numpy as np


class ReturnAsymmetryMMD_Calculator(FactorCalculator):
    """收益率分布不对称性因子（MMD版）

    基于东方证券朱剑涛研报《收益率的非对称分布与尾部蕴含的Alpha》（2021-12-25），
    原始思想来自2007年国外研究（熵原理）。

    对每只股票，在滚动窗口内计算日收益率的分布与其镜像分布（-rtn + 2*mu）的
    最大平均偏差（MMD），度量收益率分布的不对称程度。

    MMD 越大 → 收益率分布越不对称。
    """

    def __init__(self, s, lookback_days=20):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        
        close = data.xs("close_hfq", level=0)
        rtn = close.pct_change()

        min_periods = max(8, self.lookback_days // 2)
        factor = rtn.rolling(
            window=self.lookback_days, min_periods=min_periods
        ).apply(self._mmd_asymmetry, raw=True)

        return factor

    # ── MMD computation ──────────────────────────────────────────────

    @staticmethod
    def _mmd_asymmetry(x: np.ndarray) -> float:
        """Compute MMD between return series x and its mirror -x+2*mean.

        Uses data-adaptive RBF kernel bandwidth (median heuristic) so the
        factor value is meaningful regardless of the return scale.
        """
        x = x[~np.isnan(x)]
        if len(x) < 8:
            return np.nan

        mu = x.mean()
        y = -x + 2.0 * mu

        xx = x.reshape(-1, 1)
        yy = y.reshape(-1, 1)

        sq_dists_xx = (xx - xx.T) ** 2
        triu = sq_dists_xx[np.triu_indices_from(sq_dists_xx, k=1)]
        if len(triu) == 0:
            return 0.0
        pos = triu[triu > 1e-15]
        if len(pos) == 0:
            return 0.0
        gamma = 1.0 / np.median(pos)

        k_xx = np.exp(-gamma * sq_dists_xx).mean()
        k_yy = np.exp(-gamma * (yy - yy.T) ** 2).mean()
        k_xy = np.exp(-gamma * (xx - yy.T) ** 2).mean()

        return k_xx + k_yy - 2.0 * k_xy


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "收益率不对称性_MMD",
    "creator": "zhangyudong",
    "create_date": "2026.06.03",
    "formula": "MMD(rtn_window, -rtn_window + 2*mean(rtn_window)), RBF kernel gamma=1.0",
    "description": "在滚动窗口（默认20日）内计算日收益率分布与其镜像分布的MMD值，度量收益率不对称性。基于东方证券朱剑涛研报（2021-12-25），原始思想来自2007年国外研究",
    "lookback_days": 20,
    "method": "ReturnAsymmetryMMD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "反转因子"
}
'''
