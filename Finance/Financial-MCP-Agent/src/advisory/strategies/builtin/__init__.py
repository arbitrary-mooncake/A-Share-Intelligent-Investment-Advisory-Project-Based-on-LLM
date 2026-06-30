"""内置策略包 — 导入所有策略文件以触发 @register_strategy 注册。

包含 20+ 个经典量化交易策略，覆盖均线、MACD、RSI、KDJ、
布林带、唐奇安通道、海龟交易、ADX、CCI、Williams %R、随机指标、
OBV、VWAP 偏离、成交量突破、动量 ROC 等。
"""
# 导入所有策略模块以触发装饰器注册
from . import ma_cross               # noqa: F401
from . import macd                    # noqa: F401
from . import rsi                     # noqa: F401
from . import kdj                     # noqa: F401
from . import boll                    # noqa: F401
from . import donchian                # noqa: F401
from . import turtle                  # noqa: F401
from . import adx_macd                # noqa: F401
from . import triple_ma              # noqa: F401
from . import ema_sma_bias           # noqa: F401
from . import cci                     # noqa: F401
from . import williams_r             # noqa: F401
from . import stochastic             # noqa: F401
from . import rsi_ma200              # noqa: F401
from . import volume_breakout        # noqa: F401
from . import obv_cross              # noqa: F401
from . import vwap_deviation         # noqa: F401
from . import ma_cross_atr_stop      # noqa: F401
from . import vol_target_ma_cross    # noqa: F401
from . import kelly_ma_cross         # noqa: F401
from . import momentum_roc           # noqa: F401

__all__ = [
    "ma_cross",
    "macd",
    "rsi",
    "kdj",
    "boll",
    "donchian",
    "turtle",
    "adx_macd",
    "triple_ma",
    "ema_sma_bias",
    "cci",
    "williams_r",
    "stochastic",
    "rsi_ma200",
    "volume_breakout",
    "obv_cross",
    "vwap_deviation",
    "ma_cross_atr_stop",
    "vol_target_ma_cross",
    "kelly_ma_cross",
    "momentum_roc",
]
