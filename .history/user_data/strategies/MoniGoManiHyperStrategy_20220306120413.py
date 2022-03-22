# -*- coding: utf-8 -*-
# -* vim: syntax=python -*-
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# --- ↑↓ Do not remove these libs ↑↓ -----------------------------------------------------------------------------------
import sys
from pathlib import Path

import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.constants import ListPairsWithTimeframes


# Master Framework file must reside in same folder as Strategy file
sys.path.append(str(Path(__file__).parent))
from MasterMoniGoManiHyperStrategy import MasterMoniGoManiHyperStrategy
# ---- ↑ Do not remove these libs ↑ ------------------------------------------------------------------------------------
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter,merge_informative_pair
from freqtrade.data.dataprovider import DataProvider as dp 
from technical.indicators import *
from technical.util import resample_to_interval, resampled_merge
import datetime


# Define the Weighted Buy Signals to be used by MGM
buy_signals = {
    # Weighted Buy Signal: MACD above Signal
    'macd': lambda df: (df['macd'] > df['macdsignal']),
    'macdfix': lambda df: (df['buy_macdfix_macd'] > df['buy_macdfix_signal']),
    # Weighted Buy Signal: MFI crosses above 20 (Under-bought / low-price and rising indication)
    'mfi': lambda df: (qtpylib.crossed_above(df['mfi'], 20)),
    # Weighted Buy Signal: Rolling VWAP crosses above current price
    'rolling_vwap_cross': lambda df: (qtpylib.crossed_above(df['buy_rolling_vwap'], df['close'])),
    # Weighted Buy Signal: Price crosses above Parabolic SAR
    'sar_cross': lambda df: (qtpylib.crossed_above(df['buy_sar'], df['close'])),
    # Weighted Buy Signal: Stochastic Slow below 20 (Under-bought, indication of starting to move up)
    'stoch': lambda df: (df['buy_slow_k'] < 20),
    # Weighted Buy Signal: SMA long term Golden Cross (Medium term SMA crosses above Long term SMA)
    'sma_long_golden_cross': lambda df: (qtpylib.crossed_above(df['sma50'], df['sma200'])),
    'sma_long_golden_cross2': lambda df: (qtpylib.crossed_above(df['sma100'], df['sma400'])),
    # Weighted Buy Signal: SMA short term Golden Cross (Short term SMA crosses above Medium term SMA)
    'sma_short_golden_cross': lambda df: (qtpylib.crossed_above(df['sma9'], df['sma50'])),
    'sma_short_golden_cross2': lambda df: (qtpylib.crossed_above(df['sma18'], df['sma100'])),
    # Weighted Buy Signal: TEMA
    'tema': lambda df: (df['tema'] <= df['bb_middleband']) & (df['tema'] > df['tema'].shift(1)),
    'buy_boll_lower': lambda df: (df[MoniGoManiHyperStrategy.buy_boll_triger.value]<df['buy_boll_lower'] ) ,
    'buy_min': lambda df: (df['low']<df['buy_min']),
    'rsi': lambda df: (df['buy_rsi']< MoniGoManiHyperStrategy.buy_rsi.value),
    'buy_ma1': lambda df: (df['close']< df['buy_ma1']),
    'buy_ma2': lambda df: (df['close']> df['buy_ma2']),
    'buy_ma3': lambda df: (df['buy_under_ma']< df['buy_ma3']),
    'buy_env': lambda df: (df['close']< df['sma9']*MoniGoManiHyperStrategy.buy_env.value),
    'buy_cci': lambda df: (df['buy_cci']<MoniGoManiHyperStrategy.buy_cci.value),
    'cdlhammer': lambda df: ((df['cdlhammer']==100)),  
    'cdlinvertedhammer': lambda df: ((df['cdlinvertedhammer']==100)),  
    'cdldragonflydoji': lambda df: ((df['cdldragonflydoji']==100)),  
    # 'cdlpiercing': lambda df: ((df['cdlpiercing']==100)),  
    # 'cdlmorningstar': lambda df: ((df['cdlmorningstar']==100)),  
    # 'cdl3whitesoldiers': lambda df: ((df['cdl3whitesoldiers']==100)),  
    'pmax': lambda df: (df['buy_pmax']=="up"),
    'vidya': lambda df: (qtpylib.crossed_above(df['close'], df['buy_vidya'])),
    'tke': lambda df: (df['buy_tke']< 20),
    'ssl': lambda df: (df['buy_ssl_up'] > df['buy_ssl_down']),
    'vpci': lambda df: (df['buy_vpci']< MoniGoManiHyperStrategy.buy_vpci.value),
    'vfi': lambda df: ( (qtpylib.crossed_above( df['buy_vfi_1h'], 0 )) & (df["buy_vfi_1h"] > df["buy_vfima_1h"]) ),
    # # Weighted Buy Signal: MACD above Signal
    # 'macd_1h': lambda df: (df['macd_1h'] > df['macdsignal_1h']),
    # 'macdfix_1h': lambda df: (df['buy_macdfix_macd_1h'] > df['buy_macdfix_signal_1h']),
    # # Weighted Buy Signal: MFI crosses above 20 (Under-bought / low-price and rising indication)
    # 'mfi_1h': lambda df: (qtpylib.crossed_above(df['mfi_1h'], 20)),
    # # Weighted Buy Signal: Rolling VWAP crosses above current price
    # 'rolling_vwap_cross_1h': lambda df: (qtpylib.crossed_above(df['buy_rolling_vwap_1h'], df['close'])),
    # # Weighted Buy Signal: Price crosses above Parabolic SAR
    # 'sar_cross_1h': lambda df: (qtpylib.crossed_above(df['buy_sar_1h'], df['close'])),
    # # Weighted Buy Signal: Stochastic Slow below 20 (Under-bought, indication of starting to move up)
    # 'stoch_1h': lambda df: (df['buy_slow_k_1h'] < 20),
    # # Weighted Buy Signal: SMA long term Golden Cross (Medium term SMA crosses above Long term SMA)
    # 'sma_long_golden_cross_1h': lambda df: (qtpylib.crossed_above(df['sma50_1h'], df['sma200_1h'])),
    # 'sma_long_golden_cross2_1h': lambda df: (qtpylib.crossed_above(df['sma100_1h'], df['sma400_1h'])),
    # # Weighted Buy Signal: SMA short term Golden Cross (Short term SMA crosses above Medium term SMA)
    # 'sma_short_golden_cross_1h': lambda df: (qtpylib.crossed_above(df['sma9_1h'], df['sma50_1h'])),
    # 'sma_short_golden_cross2_1h': lambda df: (qtpylib.crossed_above(df['sma18_1h'], df['sma100_1h'])),
    # # Weighted Buy Signal: TEMA
    # 'tema_1h': lambda df: (df['tema_1h'] <= df['bb_middleband_1h']) & (df['tema_1h'] > df['tema_1h'].shift(1)),
    # 'buy_boll_lower': lambda df: (df[MoniGoManiHyperStrategy.buy_boll_triger.value]<df['buy_boll_lower'] ) ,
    # 'buy_min_1h': lambda df: (df['low']<df['buy_min_1h']),
    # 'rsi_1h': lambda df: (df['buy_rsi_1h']< MoniGoManiHyperStrategy.buy_rsi_1h.value),
    # 'buy_ma1_1h': lambda df: (df['close']< df['buy_ma1_1h']),
    # 'buy_env_1h': lambda df: (df['close']< df['sma9_1h']*MoniGoManiHyperStrategy.buy_env_1h.value),
    # 'buy_cci_1h': lambda df: (df['buy_cci_1h']<MoniGoManiHyperStrategy.buy_cci_1h.value),
    # # 'cdlhammer': lambda df: ((df['cdlhammer']==100)),  
    # # 'cdlinvertedhammer': lambda df: ((df['cdlinvertedhammer']==100)),  
    # # 'cdldragonflydoji': lambda df: ((df['cdldragonflydoji']==100)),  
    # # 'cdlpiercing': lambda df: ((df['cdlpiercing']==100)),  
    # # 'cdlmorningstar': lambda df: ((df['cdlmorningstar']==100)),  
    # # 'cdl3whitesoldiers': lambda df: ((df['cdl3whitesoldiers']==100)),  
    # 'pmax_1h': lambda df: (df['buy_pmax_1h']=="up"),
    # 'vidya_1h': lambda df: (qtpylib.crossed_above(df['close'], df['buy_vidya_1h'])),
    # 'tke_1h': lambda df: (df['buy_tke_1h']< 20),
    # 'ssl_1h': lambda df: (df['buy_ssl_up_1h'] > df['buy_ssl_down_1h']),
    # 'vpci_1h': lambda df: (df['buy_vpci_1h']< MoniGoManiHyperStrategy.buy_vpci_1h.value),
    # 'vfi_1h': lambda df: ( (qtpylib.crossed_above( df['buy_vfi_1h'], 0 )) & (df["buy_vfi_1h"] > df["buy_vfima_1h"]) ),
    # # Weighted Buy Signal: MACD above Signal
    # 'macd_30m': lambda df: (df['macd_30m'] > df['macdsignal_30m']),
    # 'macdfix_30m': lambda df: (df['buy_macdfix_macd_30m'] > df['buy_macdfix_signal_30m']),
    # # Weighted Buy Signal: MFI crosses above 20 (Under-bought / low-price and rising indication)
    # 'mfi_30m': lambda df: (qtpylib.crossed_above(df['mfi_30m'], 20)),
    # # Weighted Buy Signal: Rolling VWAP crosses above current price
    # 'rolling_vwap_cross_30m': lambda df: (qtpylib.crossed_above(df['buy_rolling_vwap_30m'], df['close'])),
    # # Weighted Buy Signal: Price crosses above Parabolic SAR
    # 'sar_cross_30m': lambda df: (qtpylib.crossed_above(df['buy_sar_30m'], df['close'])),
    # # Weighted Buy Signal: Stochastic Slow below 20 (Under-bought, indication of starting to move up)
    # 'stoch_30m': lambda df: (df['buy_slow_k_30m'] < 20),
    # # Weighted Buy Signal: SMA long term Golden Cross (Medium term SMA crosses above Long term SMA)
    # 'sma_long_golden_cross_30m': lambda df: (qtpylib.crossed_above(df['sma50_30m'], df['sma200_30m'])),
    # 'sma_long_golden_cross2_30m': lambda df: (qtpylib.crossed_above(df['sma100_30m'], df['sma400_30m'])),
    # # Weighted Buy Signal: SMA short term Golden Cross (Short term SMA crosses above Medium term SMA)
    # 'sma_short_golden_cross_30m': lambda df: (qtpylib.crossed_above(df['sma9_30m'], df['sma50_30m'])),
    # 'sma_short_golden_cross2_30m': lambda df: (qtpylib.crossed_above(df['sma18_30m'], df['sma100_30m'])),
    # # Weighted Buy Signal: TEMA
    # 'tema_30m': lambda df: (df['tema_30m'] <= df['bb_middleband_30m']) & (df['tema_30m'] > df['tema_30m'].shift(1)),
    # 'buy_boll_lower': lambda df: (df[MoniGoManiHyperStrategy.buy_boll_triger.value]<df['buy_boll_lower'] ) ,
    # 'buy_min_30m': lambda df: (df['low']<df['buy_min_30m']),
    # 'rsi_30m': lambda df: (df['buy_rsi_30m']< MoniGoManiHyperStrategy.buy_rsi_30m.value),
    # 'buy_ma1_30m': lambda df: (df['close']< df['buy_ma1_30m']),
    # 'buy_env_30m': lambda df: (df['close']< df['sma9_30m']*MoniGoManiHyperStrategy.buy_env_30m.value),
    # 'buy_cci_30m': lambda df: (df['buy_cci_30m']<MoniGoManiHyperStrategy.buy_cci_30m.value),
    # # 'cdlhammer': lambda df: ((df['cdlhammer']==100)),  
    # # 'cdlinvertedhammer': lambda df: ((df['cdlinvertedhammer']==100)),  
    # # 'cdldragonflydoji': lambda df: ((df['cdldragonflydoji']==100)),  
    # # 'cdlpiercing': lambda df: ((df['cdlpiercing']==100)),  
    # # 'cdlmorningstar': lambda df: ((df['cdlmorningstar']==100)),  
    # # 'cdl3whitesoldiers': lambda df: ((df['cdl3whitesoldiers']==100)),  
    # 'pmax_30m': lambda df: (df['buy_pmax_30m']=="up"),
    # 'vidya_30m': lambda df: (qtpylib.crossed_above(df['close'], df['buy_vidya_30m'])),
    # 'tke_30m': lambda df: (df['buy_tke_30m']< 20),
    # 'ssl_30m': lambda df: (df['buy_ssl_up_30m'] > df['buy_ssl_down_30m']),
    # 'vpci_30m': lambda df: (df['buy_vpci_30m']< MoniGoManiHyperStrategy.buy_vpci_30m.value),
    # 'vfi_30m': lambda df: ( (qtpylib.crossed_above( df['buy_vfi_30m'], 0 )) & (df["buy_vfi_30m"] > df["buy_vfima_30m"]) ),

}

# Define the Weighted Sell Signals to be used by MGM
sell_signals = {
    # Weighted Sell Signal: MACD below Signal
    'macd': lambda df: (df['macd'] < df['macdsignal']),
    'macdfix': lambda df: (df['sell_macdfix_macd'] < df['sell_macdfix_signal']),
    # Weighted Sell Signal: MFI crosses below 80 (Over-bought / high-price and dropping indication)
    'mfi': lambda df: (qtpylib.crossed_below(df['mfi'], 80)),
    # Weighted Sell Signal: Rolling VWAP crosses below current price
    'rolling_vwap_cross': lambda df: (qtpylib.crossed_below(df['sell_rolling_vwap'], df['close'])),
    # Weighted Sell Signal: Price crosses below Parabolic SAR
    'sar_cross': lambda df: (qtpylib.crossed_below(df['sell_sar'], df['close'])),
    # Weighted Sell Signal: Stochastic Slow above 80 (Over-bought, indication of starting to move down)
    'stoch': lambda df: (df['sell_slow_k'] > 80),
    # Weighted Sell Signal: SMA long term Death Cross (Medium term SMA crosses below Long term SMA)
    'sma_long_death_cross': lambda df: (qtpylib.crossed_below(df['sma50'], df['sma200'])),
    'sma_long_death_cross2': lambda df: (qtpylib.crossed_below(df['sma100'], df['sma400'])),
    # Weighted Sell Signal: SMA short term Death Cross (Short term SMA crosses below Medium term SMA)
    'sma_short_death_cross': lambda df: (qtpylib.crossed_below(df['sma9'], df['sma50'])),
    'sma_short_death_cross2': lambda df: (qtpylib.crossed_below(df['sma18'], df['sma100'])),
    # Weighted Buy Signal: TEMA
    'tema': lambda df: (df['tema'] > df['bb_middleband']) & (df['tema'] < df['tema'].shift(1)),
    'sell_boll_upper': lambda df: (df[MoniGoManiHyperStrategy.sell_boll_triger.value]>df['sell_boll_upper']) ,
    'sell_max': lambda df: (df['high']> df['sell_max']),
    'rsi': lambda df: (df['sell_rsi']> MoniGoManiHyperStrategy.sell_rsi.value),
    'sell_ma1': lambda df: (df['close']> df['sell_ma1']),
    'sell_ma2': lambda df: (df['close']< df['sell_ma2']),
    'sell_ma3': lambda df: (df['sell_upper_ma']> df['sell_ma3']),
    'sell_env': lambda df: (df['close']> df['sma9']*MoniGoManiHyperStrategy.sell_env.value),
    'sell_cci': lambda df: (df['sell_cci']>MoniGoManiHyperStrategy.buy_cci.value),
    'cdlhangingman': lambda df: ((df['cdlhangingman']==100)),  
    'cdlshootingstar': lambda df: ((df['cdlshootingstar']==100)),  
    # 'cdlgravestonedoji': lambda df: ((df['cdlgravestonedoji']==100)),  
    # 'cdldarkcloudcover': lambda df: ((df['cdldarkcloudcover']==100)),  
    # 'cdleveningdojistar': lambda df: ((df['cdleveningdojistar']==100)),  
    # 'cdleveningstar': lambda df: ((df['cdleveningstar']==100)),  
    'pmax': lambda df: (df['sell_pmax']=="down"),
    'vidya': lambda df: (qtpylib.crossed_below(df['close'], df['buy_vidya'])),
    'tke': lambda df: (df['sell_tke']> 80),
    'ssl': lambda df: (df['sell_ssl_up'] < df['sell_ssl_down']),
    'vpci': lambda df: (df['buy_vpci']> MoniGoManiHyperStrategy.sell_vpci.value),
    'vfi': lambda df: ((qtpylib.crossed_below(df['sell_vfi_1h'],0)) & (df["sell_vfi_1h"] < df["sell_vfima_1h"])),
    # # Weighted Sell Signal: MACD below Signal
    # 'macd_1h': lambda df: (df['macd_1h'] < df['macdsignal_1h']),
    # 'macdfix_1h': lambda df: (df['sell_macdfix_macd_1h'] < df['sell_macdfix_signal_1h']),
    # # Weighted Sell Signal: MFI crosses below 80 (Over-bought / high-price and dropping indication)
    # 'mfi_1h': lambda df: (qtpylib.crossed_below(df['mfi_1h'], 80)),
    # # Weighted Sell Signal: Rolling VWAP crosses below current price
    # 'rolling_vwap_cross_1h': lambda df: (qtpylib.crossed_below(df['sell_rolling_vwap_1h'], df['close'])),
    # # Weighted Sell Signal: Price crosses below Parabolic SAR
    # 'sar_cross_1h': lambda df: (qtpylib.crossed_below(df['sell_sar_1h'], df['close'])),
    # # Weighted Sell Signal: Stochastic Slow above 80 (Over-bought, indication of starting to move down)
    # 'stoch_1h': lambda df: (df['sell_slow_k_1h'] > 80),
    # # Weighted Sell Signal: SMA long term Death Cross (Medium term SMA crosses below Long term SMA)
    # 'sma_long_death_cross_1h': lambda df: (qtpylib.crossed_below(df['sma50_1h'], df['sma200_1h'])),
    # 'sma_long_death_cross2_1h': lambda df: (qtpylib.crossed_below(df['sma100_1h'], df['sma400_1h'])),
    # # Weighted Sell Signal: SMA short term Death Cross (Short term SMA crosses below Medium term SMA)
    # 'sma_short_death_cross_1h': lambda df: (qtpylib.crossed_below(df['sma9_1h'], df['sma50_1h'])),
    # 'sma_short_death_cross2_1h': lambda df: (qtpylib.crossed_below(df['sma18_1h'], df['sma100_1h'])),
    # # Weighted Buy Signal: TEMA
    # 'tema_1h': lambda df: (df['tema_1h'] > df['bb_middleband_1h']) & (df['tema_1h'] < df['tema_1h'].shift(1)),
    # 'sell_boll_upper_1h': lambda df: (df[MoniGoManiHyperStrategy.sell_boll_triger_1h.value]>df['sell_boll_upper_1h']) ,
    # 'sell_max_1h': lambda df: (df['high']> df['sell_max_1h']),
    # 'rsi_1h': lambda df: (df['sell_rsi_1h']> MoniGoManiHyperStrategy.sell_rsi_1h.value),
    # 'sell_ma1_1h': lambda df: (df['close']> df['sell_ma1_1h']),
    # 'sell_env_1h': lambda df: (df['close']> df['sma9_1h']*MoniGoManiHyperStrategy.sell_env_1h.value),
    # 'sell_cci_1h': lambda df: (df['sell_cci_1h']>MoniGoManiHyperStrategy.buy_cci_1h.value),
    # # 'cdlhangingman': lambda df: ((df['cdlhangingman']==100)),  
    # # 'cdlshootingstar': lambda df: ((df['cdlshootingstar']==100)),  
    # # 'cdlgravestonedoji': lambda df: ((df['cdlgravestonedoji']==100)),  
    # # 'cdldarkcloudcover': lambda df: ((df['cdldarkcloudcover']==100)),  
    # # 'cdleveningdojistar': lambda df: ((df['cdleveningdojistar']==100)),  
    # # 'cdleveningstar': lambda df: ((df['cdleveningstar']==100)),  
    # 'pmax_1h': lambda df: (df['sell_pmax_1h']=="down"),
    # 'vidya_1h': lambda df: (qtpylib.crossed_below(df['close'], df['buy_vidya_1h'])),
    # 'tke_1h': lambda df: (df['sell_tke_1h']> 80),
    # 'ssl_1h': lambda df: (df['sell_ssl_up_1h'] < df['sell_ssl_down_1h']),
    # 'vpci_1h': lambda df: (df['buy_vpci']> MoniGoManiHyperStrategy.sell_vpci_1h.value),
    # 'vfi_1h': lambda df: ((qtpylib.crossed_below(df['sell_vfi_1h'],0)) & (df["sell_vfi_1h"] < df["sell_vfima_1h"])),
    # # Weighted Sell Signal: MACD below Signal
    # 'macd_30m': lambda df: (df['macd_30m'] < df['macdsignal_30m']),
    # 'macdfix_30m': lambda df: (df['sell_macdfix_macd_30m'] < df['sell_macdfix_signal_30m']),
    # # Weighted Sell Signal: MFI crosses below 80 (Over-bought / high-price and dropping indication)
    # 'mfi_30m': lambda df: (qtpylib.crossed_below(df['mfi_30m'], 80)),
    # # Weighted Sell Signal: Rolling VWAP crosses below current price
    # 'rolling_vwap_cross_30m': lambda df: (qtpylib.crossed_below(df['sell_rolling_vwap_30m'], df['close'])),
    # # Weighted Sell Signal: Price crosses below Parabolic SAR
    # 'sar_cross_30m': lambda df: (qtpylib.crossed_below(df['sell_sar_30m'], df['close'])),
    # # Weighted Sell Signal: Stochastic Slow above 80 (Over-bought, indication of starting to move down)
    # 'stoch_30m': lambda df: (df['sell_slow_k_30m'] > 80),
    # # Weighted Sell Signal: SMA long term Death Cross (Medium term SMA crosses below Long term SMA)
    # 'sma_long_death_cross_30m': lambda df: (qtpylib.crossed_below(df['sma50_30m'], df['sma200_30m'])),
    # 'sma_long_death_cross2_30m': lambda df: (qtpylib.crossed_below(df['sma100_30m'], df['sma400_30m'])),
    # # Weighted Sell Signal: SMA short term Death Cross (Short term SMA crosses below Medium term SMA)
    # 'sma_short_death_cross_30m': lambda df: (qtpylib.crossed_below(df['sma9_30m'], df['sma50_30m'])),
    # 'sma_short_death_cross2_30m': lambda df: (qtpylib.crossed_below(df['sma18_30m'], df['sma100_30m'])),
    # # Weighted Buy Signal: TEMA
    # 'tema_30m': lambda df: (df['tema_30m'] > df['bb_middleband_30m']) & (df['tema_30m'] < df['tema_30m'].shift(1)),
    # 'sell_boll_upper_30m': lambda df: (df[MoniGoManiHyperStrategy.sell_boll_triger_30m.value]>df['sell_boll_upper_30m']) ,
    # 'sell_max_30m': lambda df: (df['high']> df['sell_max_30m']),
    # 'rsi_30m': lambda df: (df['sell_rsi_30m']> MoniGoManiHyperStrategy.sell_rsi_30m.value),
    # 'sell_ma1_30m': lambda df: (df['close']> df['sell_ma1_30m']),
    # 'sell_env_30m': lambda df: (df['close']> df['sma9_30m']*MoniGoManiHyperStrategy.sell_env_30m.value),
    # 'sell_cci_30m': lambda df: (df['sell_cci_30m']>MoniGoManiHyperStrategy.buy_cci_30m.value),
    # # 'cdlhangingman': lambda df: ((df['cdlhangingman']==100)),  
    # # 'cdlshootingstar': lambda df: ((df['cdlshootingstar']==100)),  
    # # 'cdlgravestonedoji': lambda df: ((df['cdlgravestonedoji']==100)),  
    # # 'cdldarkcloudcover': lambda df: ((df['cdldarkcloudcover']==100)),  
    # # 'cdleveningdojistar': lambda df: ((df['cdleveningdojistar']==100)),  
    # # 'cdleveningstar': lambda df: ((df['cdleveningstar']==100)),  
    # 'pmax_30m': lambda df: (df['sell_pmax_30m']=="down"),
    # 'vidya_30m': lambda df: (qtpylib.crossed_below(df['close'], df['buy_vidya_30m'])),
    # 'tke_30m': lambda df: (df['sell_tke_30m']> 80),
    # 'ssl_30m': lambda df: (df['sell_ssl_up_30m'] < df['sell_ssl_down_30m']),
    # 'vpci_30m': lambda df: (df['buy_vpci']> MoniGoManiHyperStrategy.sell_vpci_30m.value),
    # 'vfi_30m': lambda df: ((qtpylib.crossed_below(df['sell_vfi_30m'],0)) & (df["sell_vfi_30m"] < df["sell_vfima_30m"])),






}


# Returns the method responsible for decorating the current class with all the parameters of the MGM
generate_mgm_attributes = MasterMoniGoManiHyperStrategy.generate_mgm_attributes(buy_signals, sell_signals)


@generate_mgm_attributes
class MoniGoManiHyperStrategy(MasterMoniGoManiHyperStrategy):
    seq= [5, 990]

    """
    add_buy_signals
    """

    buy_boll_period = IntParameter(low=5, high=seq[-1], default=443, space="buy")
    buy_boll_stds = DecimalParameter(0.1, 5.0, default=4.598, space="buy")
    buy_boll_matype = IntParameter(0, 8, default=2, space="buy")
    buy_boll_triger = CategoricalParameter(["close", "low"], space="buy")

    buy_min_period = IntParameter(low=seq[0], high=seq[-1], default=402, space="buy")

    buy_rsi = IntParameter(low=10, high=50, default=30, space="buy")
    buy_rsi_period = IntParameter(low=5, high=990, default=30, space="buy")


    buy_ma1_type = IntParameter(0, 8, default=0, space="buy", load=True, optimize=True)
    buy_ma1_period = IntParameter(low=seq[0], high=seq[-1],
                                  default=402, space="buy", load=True, optimize=True)
    buy_ma2_type = IntParameter(0, 8, default=0, space="buy", load=True, optimize=True)
    buy_ma2_period = IntParameter(low=seq[0], high=seq[-1],
                                  default=402, space="buy", load=True, optimize=True)

    buy_env= DecimalParameter(0.8,0.99,default=0.99,space="buy")

    buy_cci=IntParameter(5,50,default=50,space="buy")

    buy_sar_period = IntParameter(low=5, high=990, default=30, space="buy")

    buy_roll_period = IntParameter(low=5, high=990, default=30, space="buy")

    buy_stoch_period = IntParameter(low=5, high=990, default=30, space="buy")

    buy_macdfix_period = IntParameter(low=5, high=990, default=30, space="buy")
    buy_cdlhammer_length = IntParameter(low=2, high=48, default=2, space="buy")
    buy_cdlinvertedhammer_length = IntParameter(low=2, high=48, default=2, space="buy")
    # buy_cdldragonflydoji_length = IntParameter(low=2, high=48, default=2, space="buy")
    # buy_cdlpiercing_length = IntParameter(low=2, high=48, default=2, space="buy")
    # buy_cdlmorningstar_length = IntParameter(low=2, high=48, default=2, space="buy")
    # buy_cdl3whitesoldiers_length = IntParameter(low=2, high=48, default=2, space="buy")

    buy_vidya_length = IntParameter(low=9, high=990, default=9, space="buy")
    
    buy_tke_length = IntParameter(low=9, high=990, default=9, space="buy")

    buy_cci_period = IntParameter(low=9, high=990, default=9, space="buy")

    buy_pmax_period = IntParameter(low=5, high=990, default=876, space="buy")

    buy_ssl_length = IntParameter(low=9, high=990, default=9, space="buy")

    buy_vpci_length = IntParameter(low=5, high=990, default=9, space="buy")
    buy_vpci=IntParameter(-50,0,default=-25,space="buy")

    buy_vfi_length = IntParameter(low=5, high=990, default=9, space="buy")

    buy_ma3_type = IntParameter(0, 8, default=0, space="buy", load=True, optimize=True)
    buy_ma3_period = IntParameter(low=seq[0], high=seq[-1],
                                  default=402, space="buy", load=True, optimize=True)
    buy_under_ma_period = IntParameter(low=10, high=100, default=9, space="buy")

    # buy_boll_period_1h = IntParameter(low=5, high=seq[-1], default=443, space="buy")
    # buy_boll_stds_1h = DecimalParameter(0.1, 5.0, default=4.598, space="buy")
    # buy_boll_matype_1h = IntParameter(0, 8, default=2, space="buy")
    # buy_boll_triger_1h = CategoricalParameter(["close", "low"], space="buy")

    # buy_min_period_1h = IntParameter(low=seq[0], high=seq[-1], default=402, space="buy")

    # buy_rsi_1h = IntParameter(low=10, high=50, default=30, space="buy")
    # buy_rsi_period_1h = IntParameter(low=5, high=990, default=30, space="buy")


    # buy_ma1_type_1h = IntParameter(0, 8, default=0, space="buy", load=True, optimize=True)
    # buy_ma1_period_1h = IntParameter(low=seq[0], high=seq[-1],
    #                               default=402, space="buy", load=True, optimize=True)

    # buy_env_1h= DecimalParameter(0.8,0.99,default=0.99,space="buy")

    # buy_cci_1h=IntParameter(5,50,default=50,space="buy")

    # buy_sar_period_1h = IntParameter(low=5, high=990, default=30, space="buy")

    # buy_roll_period_1h = IntParameter(low=5, high=990, default=30, space="buy")

    # buy_stoch_period_1h = IntParameter(low=5, high=990, default=30, space="buy")

    # buy_macdfix_period_1h = IntParameter(low=5, high=990, default=30, space="buy")

    # buy_vidya_length_1h = IntParameter(low=9, high=990, default=9, space="buy")
    
    # buy_tke_length_1h = IntParameter(low=9, high=990, default=9, space="buy")

    # buy_cci_period_1h = IntParameter(low=9, high=990, default=9, space="buy")

    # buy_pmax_period_1h = IntParameter(low=5, high=990, default=876, space="buy")

    # buy_ssl_length_1h = IntParameter(low=9, high=990, default=9, space="buy")

    # buy_vpci_length_1h = IntParameter(low=5, high=990, default=9, space="buy")
    # buy_vpci_1h=IntParameter(-50,0,default=-25,space="buy")

    # buy_vfi_length_1h = IntParameter(low=5, high=990, default=9, space="buy")


    # buy_boll_period_30m = IntParameter(low=5, high=seq[-1], default=443, space="buy")
    # buy_boll_stds_30m = DecimalParameter(0.1, 5.0, default=4.598, space="buy")
    # buy_boll_matype_30m = IntParameter(0, 8, default=2, space="buy")
    # buy_boll_triger_30m = CategoricalParameter(["close", "low"], space="buy")

    # buy_min_period_30m = IntParameter(low=seq[0], high=seq[-1], default=402, space="buy")

    # buy_rsi_30m = IntParameter(low=10, high=50, default=30, space="buy")
    # buy_rsi_period_30m = IntParameter(low=5, high=990, default=30, space="buy")


    # buy_ma1_type_30m = IntParameter(0, 8, default=0, space="buy", load=True, optimize=True)
    # buy_ma1_period_30m = IntParameter(low=seq[0], high=seq[-1],
    #                               default=402, space="buy", load=True, optimize=True)

    # buy_env_30m= DecimalParameter(0.8,0.99,default=0.99,space="buy")

    # buy_cci_30m=IntParameter(5,50,default=50,space="buy")

    # buy_sar_period_30m = IntParameter(low=5, high=990, default=30, space="buy")

    # buy_roll_period_30m = IntParameter(low=5, high=990, default=30, space="buy")

    # buy_stoch_period_30m = IntParameter(low=5, high=990, default=30, space="buy")

    # buy_macdfix_period_30m = IntParameter(low=5, high=990, default=30, space="buy")

    # buy_vidya_length_30m = IntParameter(low=9, high=990, default=9, space="buy")
    
    # buy_tke_length_30m = IntParameter(low=9, high=990, default=9, space="buy")

    # buy_cci_period_30m = IntParameter(low=9, high=990, default=9, space="buy")

    # buy_pmax_period_30m = IntParameter(low=5, high=990, default=876, space="buy")

    # buy_ssl_length_30m = IntParameter(low=9, high=990, default=9, space="buy")

    # buy_vpci_length_30m = IntParameter(low=5, high=990, default=9, space="buy")
    # buy_vpci_30m=IntParameter(-50,0,default=-25,space="buy")

    # buy_vfi_length_30m = IntParameter(low=5, high=990, default=9, space="buy")

    """
    add_sell_signals
    """
    sell_boll_period = IntParameter(low=5, high=seq[-1], default=402, space="sell")
    sell_boll_stds = DecimalParameter(0.1, 5.0, default=2.0, space="sell")
    sell_boll_matype = IntParameter(0, 8, default=0, space="sell")
    sell_boll_triger = CategoricalParameter(["close", "high"], space="sell")

    sell_max_period = IntParameter(low=seq[0], high=seq[-1], default=402, space="sell")

    sell_rsi = IntParameter(low=50, high=90, default=50, space="sell")
    sell_rsi_period = IntParameter(low=5, high=990, default=30, space="sell")


    sell_ma1_type = IntParameter(0, 8, default=0, space="sell")
    sell_ma1_period = IntParameter(low=seq[0], high=seq[-1], default=402, space="sell")
    sell_ma2_type = IntParameter(0, 8, default=0, space="sell")
    sell_ma2_period = IntParameter(low=seq[0], high=seq[-1], default=402, space="sell")

    sell_env= DecimalParameter(1.01,1.2,default=1.01,space="sell")

    sell_cci=IntParameter(50,90,default=50,space="sell")

    sell_sar_period = IntParameter(low=5, high=990, default=30, space="sell")

    sell_roll_period = IntParameter(low=5, high=990, default=30, space="sell")

    sell_stoch_period = IntParameter(low=5, high=990, default=30, space="sell")

    sell_macdfix_period = IntParameter(low=5, high=990, default=30, space="sell")

    sell_cdlhangingman_length = IntParameter(low=2, high=48, default=2, space="sell")
    sell_cdlshootingstar_length = IntParameter(low=2, high=48, default=2, space="sell")
    # sell_cdlgravestonedoji_length = IntParameter(low=2, high=48, default=2, space="sell")
    # sell_cdldarkcloudcover_length = IntParameter(low=2, high=48, default=2, space="sell")
    # sell_cdleveningdojistar_length = IntParameter(low=2, high=48, default=2, space="sell")
    # sell_cdleveningstar_length = IntParameter(low=2, high=48, default=2, space="sell")

    sell_vidya_length = IntParameter(low=9, high=990, default=9, space="sell")

    sell_tke_length = IntParameter(low=9, high=990, default=9, space="sell")

    sell_cci_period = IntParameter(low=9, high=990, default=9, space="sell")

    sell_pmax_period = IntParameter(low=5, high=990, default=876, space="sell")

    sell_ssl_length = IntParameter(low=9, high=990, default=9, space="sell")

    sell_vpci_length = IntParameter(low=5, high=990, default=9, space="sell")
    sell_vpci=IntParameter(0,50,default=25,space="sell")

    sell_vfi_length = IntParameter(low=5, high=990, default=9, space="sell")

    sell_ma3_type = IntParameter(0, 8, default=0, space="sell", load=True, optimize=True)
    sell_ma3_period = IntParameter(low=seq[0], high=seq[-1],
                                  default=402, space="sell", load=True, optimize=True)
    sell_upper_ma_period = IntParameter(low=10, high=100, default=9, space="sell")

    # sell_boll_period_1h = IntParameter(low=5, high=seq[-1], default=402, space="sell")
    # sell_boll_stds_1h = DecimalParameter(0.1, 5.0, default=2.0, space="sell")
    # sell_boll_matype_1h = IntParameter(0, 8, default=0, space="sell")
    # sell_boll_triger_1h = CategoricalParameter(["close", "high"], space="sell")

    # sell_max_period_1h = IntParameter(low=seq[0], high=seq[-1], default=402, space="sell")

    # sell_rsi_1h = IntParameter(low=50, high=90, default=50, space="sell")
    # sell_rsi_period_1h = IntParameter(low=5, high=990, default=30, space="sell")


    # sell_ma1_type_1h = IntParameter(0, 8, default=0, space="sell")
    # sell_ma1_period_1h = IntParameter(low=seq[0], high=seq[-1], default=402, space="sell")

    # sell_env_1h= DecimalParameter(1.01,1.2,default=1.01,space="sell")

    # sell_cci_1h=IntParameter(50,90,default=50,space="sell")

    # sell_sar_period_1h = IntParameter(low=5, high=990, default=30, space="sell")

    # sell_roll_period_1h = IntParameter(low=5, high=990, default=30, space="sell")

    # sell_stoch_period_1h = IntParameter(low=5, high=990, default=30, space="sell")

    # sell_macdfix_period_1h = IntParameter(low=5, high=990, default=30, space="sell")

    # # sell_cdlhangingman_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdlshootingstar_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdlgravestonedoji_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdldarkcloudcover_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdleveningdojistar_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdleveningstar_length = IntParameter(low=2, high=48, default=2, space="sell")

    # sell_vidya_length_1h = IntParameter(low=9, high=990, default=9, space="sell")

    # sell_tke_length_1h = IntParameter(low=9, high=990, default=9, space="sell")

    # sell_cci_period_1h = IntParameter(low=9, high=990, default=9, space="sell")

    # sell_pmax_period_1h = IntParameter(low=5, high=990, default=876, space="sell")

    # sell_ssl_length_1h = IntParameter(low=9, high=990, default=9, space="sell")

    # sell_vpci_length_1h = IntParameter(low=5, high=990, default=9, space="sell")
    # sell_vpci_1h=IntParameter(0,50,default=25,space="sell")

    # sell_vfi_length_1h = IntParameter(low=5, high=990, default=9, space="sell")


    # sell_boll_period_30m = IntParameter(low=5, high=seq[-1], default=402, space="sell")
    # sell_boll_stds_30m = DecimalParameter(0.1, 5.0, default=2.0, space="sell")
    # sell_boll_matype_30m = IntParameter(0, 8, default=0, space="sell")
    # sell_boll_triger_30m = CategoricalParameter(["close", "high"], space="sell")

    # sell_max_period_30m = IntParameter(low=seq[0], high=seq[-1], default=402, space="sell")

    # sell_rsi_30m = IntParameter(low=50, high=90, default=50, space="sell")
    # sell_rsi_period_30m = IntParameter(low=5, high=990, default=30, space="sell")


    # sell_ma1_type_30m = IntParameter(0, 8, default=0, space="sell")
    # sell_ma1_period_30m = IntParameter(low=seq[0], high=seq[-1], default=402, space="sell")

    # sell_env_30m= DecimalParameter(1.01,1.2,default=1.01,space="sell")

    # sell_cci_30m=IntParameter(50,90,default=50,space="sell")

    # sell_sar_period_30m = IntParameter(low=5, high=990, default=30, space="sell")

    # sell_roll_period_30m = IntParameter(low=5, high=990, default=30, space="sell")

    # sell_stoch_period_30m = IntParameter(low=5, high=990, default=30, space="sell")

    # sell_macdfix_period_30m = IntParameter(low=5, high=990, default=30, space="sell")

    # # sell_cdlhangingman_length = IntParameter(low=2, high=48, default=2, space="sell")
    sell_cdlshootingstar_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdlgravestonedoji_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdldarkcloudcover_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdleveningdojistar_length = IntParameter(low=2, high=48, default=2, space="sell")
    # # sell_cdleveningstar_length = IntParameter(low=2, high=48, default=2, space="sell")

    # sell_vidya_length_30m = IntParameter(low=9, high=990, default=9, space="sell")

    # sell_tke_length_30m = IntParameter(low=9, high=990, default=9, space="sell")

    # sell_cci_period_30m = IntParameter(low=9, high=990, default=9, space="sell")

    # sell_pmax_period_30m = IntParameter(low=5, high=990, default=876, space="sell")

    # sell_ssl_length_30m = IntParameter(low=9, high=990, default=9, space="sell")

    # sell_vpci_length_30m = IntParameter(low=5, high=990, default=9, space="sell")
    # sell_vpci_30m=IntParameter(0,50,default=25,space="sell")

    # sell_vfi_length_30m = IntParameter(low=5, high=990, default=9, space="sell")


    """
    add_share_signals
    """




    """
    ####################################################################################
    ####                                                                            ####
    ###                  MoniGoMani Base Strategy v0.13.0 by Rikj000                 ###
    ##                          -----------------------------                         ##
    #               Isn't that what we all want? Our money to go many?                 #
    #          Well that's what this Freqtrade strategy hopes to do for you!           #
    ##       By giving you/HyperOpt a lot of signals to alter the weight from         ##
    ###           ------------------------------------------------------             ###
    ##        Big thank you to xmatthias and everyone who helped on MoniGoMani,       ##
    ##      Freqtrade Discord support was also really helpful so thank you too!       ##
    ###         -------------------------------------------------------              ###
    ##              Disclaimer: This strategy is under development.                   ##
    #      I do not recommend running it live until further development/testing.       #
    ##                      TEST IT BEFORE USING IT!                                  ##
    ###                                                              ▄▄█▀▀▀▀▀█▄▄     ###
    ##               -------------------------------------         ▄█▀  ▄ ▄    ▀█▄    ##
    ###   If you like my work, feel free to donate or use one of   █   ▀█▀▀▀▀▄   █   ###
    ##   my referral links, that would also greatly be appreciated █    █▄▄▄▄▀   █    ##
    #     ICONOMI: https://www.iconomi.com/register?ref=JdFzz      █    █    █   █     #
    ##  Binance: https://www.binance.com/en/register?ref=97611461  ▀█▄ ▀▀█▀█▀  ▄█▀    ##
    ###          BTC: 19LL2LCMZo4bHJgy15q1Z1bfe7mV4bfoWK             ▀▀█▄▄▄▄▄█▀▀     ###
    ####                                                                            ####
    ####################################################################################
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the Freqtrade documentation or it's Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Plot configuration to show all Weighted Signals/Indicators used by MoniGoMani in FreqUI.
    # Also loads in MGM Framework Plots for Buy/Sell Signals/Indicators and Trend Detection.
    plot_config = MasterMoniGoManiHyperStrategy.populate_frequi_plots({
        # Main Plots Signals/Indicators (SMAs, EMAs, Bollinger Bands, Rolling VWAP, TEMA)
        'main_plot': {
            'sma9': {'color': '#2c05f6'},
            'sma50': {'color': '#19038a'},
            'sma200': {'color': '#0d043b'},
            'ema9': {'color': '#12e5a6'},
            'ema50': {'color': '#0a8963'},
            'ema200': {'color': '#074b36'},
            'bb_middleband': {'color': '#6f1a7b'},
            'rolling_vwap': {'color': '#727272'},
            'tema': {'color': '#9345ee'}
        },
        # Sub Plots - Each dict defines one additional plot
        'subplots': {
            # Sub Plots - Individual Weighted Signals/Indicators
            'ADX (Average Directional Index)': {
                'adx': {'color': '#6f1a7b'}
            },
            'MACD (Moving Average Convergence Divergence)': {
                'macd': {'color': '#19038a'},
                'macdsignal': {'color': '#ae231c'}
            },
            'MFI (Money Flow Index)': {
                'mfi': {'color': '#7fba3c'}
            },
            'RSI (Relative Strength Index)': {
                'rsi': {'color': '#7fb92a'}
            },
            'Stochastic Slow': {
                'slowk': {'color': '#14efe7'}
            }
        }
    })

    def informative_pairs(self) -> ListPairsWithTimeframes:
        """
        Defines additional informative pair/interval combinations to be cached from the exchange,
        these will be used during TimeFrame-Zoom.

        :return informative_pairs: (list) List populated with additional informative pairs
        """

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        informative_pairs += [(pair, self.core_trend_timeframe) for pair in pairs]
        informative_pairs += [(pair, "1h") for pair in pairs]
        informative_pairs += [(pair, "30m") for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds base indicators based on Run-Mode & TimeFrame-Zoom

        :param dataframe: (DataFrame) DataFrame with data from the exchange
        :param metadata: (dict) Additional information, like the currently traded pair
        :return DataFrame: DataFrame for MoniGoMani with all mandatory indicator data populated
        """

        return self._populate_indicators(dataframe=dataframe, metadata=metadata)

    def do_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to MoniGoMani's DataFrame per pair.
        Should be called with 'informative_pair' (1h candles) during backtesting/hyperopting with TimeFrame-Zoom!

        Performance Note: For the best performance be frugal on the number of indicators you are using.
        Only add in indicators that you are using in your weighted signal configuration for MoniGoMani,
        otherwise you will waste your memory and CPU usage.

        :param dataframe: (DataFrame) DataFrame with data from the exchange
        :param metadata: (dict) Additional information, like the currently traded pair
        :return DataFrame: DataFrame for MoniGoMani with all mandatory indicator data populated
        """
        info_time="1h"
        # info_time_2="30m"
        if self.dp.runmode.value in ('live', 'dry_run'):
            info=self.dp.get_pair_dataframe(pair=metadata["pair"],timeframe=info_time)
            # info_2=self.dp.get_pair_dataframe(pair=metadata["pair"],timeframe=info_time_2)
        else:
            info=self.dp.historic_ohlcv(pair=metadata["pair"],timeframe=info_time)
            # info_2=self.dp.historic_ohlcv(pair=metadata["pair"],timeframe=info_time_2)
        

        # Momentum Indicators (timeperiod is expressed in candles)
        # -------------------

        # Parabolic SAR
        # dataframe['sar'] = ta.SAR(dataframe)

        # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowk'] = stoch['slowk']

        # MACD - Moving Average Convergence Divergence
        macd = ta.MACD(dataframe)

        dataframe['macd'] = macd['macd']  # MACD - Blue TradingView Line (Bullish if on top)
        dataframe['macdsignal'] = macd['macdsignal']  # Signal - Orange TradingView Line (Bearish if on top)

        # MFI - Money Flow Index (Under bought / Over sold & Over bought / Under sold / volume Indicator)
        dataframe['mfi'] = ta.MFI(dataframe)

        # Overlap Studies
        # ---------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_middleband'] = bollinger['mid']

        # SMA's & EMA's are trend following tools (Should not be used when line goes sideways)
        # SMA - Simple Moving Average (Moves slower compared to EMA, price trend over X periods)
        dataframe['sma9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma18'] = ta.SMA(dataframe, timeperiod=18)
        dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['sma400'] = ta.SMA(dataframe, timeperiod=400)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)


        # Volume Indicators
        # -----------------

        # Rolling VWAP - Volume Weighted Average Price
        # dataframe['rolling_vwap'] = qtpylib.rolling_vwap(dataframe)

        """
        add_buy_signal
        """
        buy_boll = ta.BBANDS(dataframe, timeperiod=self.buy_boll_period.value,
                            nbdevup=float(self.buy_boll_stds.value), nbdevdn=float(self.buy_boll_stds.value), matype=self.buy_boll_matype.value)
        dataframe["buy_boll_lower"] = buy_boll["lowerband"]

        dataframe["buy_rsi"] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period.value)

        buy_min_tail = dataframe["close"].tail(self.buy_min_period.value)
        dataframe["buy_min"] = buy_min_tail.min()

        dataframe["buy_ma1"] = ta.MA(dataframe,timeperiod=self.buy_ma1_period.value,matype=self.buy_ma1_type.value)
        dataframe["buy_ma2"] = ta.MA(dataframe,timeperiod=self.buy_ma2_period.value,matype=self.buy_ma2_type.value)
        dataframe["buy_ma3"] = ta.MA(dataframe,timeperiod=self.buy_ma3_period.value,matype=self.buy_ma3_type.value)

        dataframe['buy_rolling_vwap'] = qtpylib.rolling_vwap(dataframe,window=self.buy_roll_period.value)

        buy_stoch = qtpylib.stoch(dataframe,window=self.buy_stoch_period.value)
        dataframe['buy_slow_k']=buy_stoch["slow_k"]

        dataframe['buy_sar'] = ta.SAR(dataframe,timeperiod=self.buy_sar_period.value)

        buy_macdfix=ta.MACDFIX(dataframe,signalperiod=self.buy_macdfix_period.value)
        dataframe["buy_macdfix_macd"]=buy_macdfix["macd"]
        dataframe["buy_macdfix_signal"]=buy_macdfix["macdsignal"]

        dataframe["buy_vidya"] = VIDYA(dataframe,length=self.buy_vidya_length.value)

        dataframe["buy_tke"],tke = TKE(dataframe,length=self.buy_tke_length.value)

        dataframe["buy_cci"] = ta.CCI(dataframe,timeperiod=self.buy_cci_period.value)

        
        buy_pmax = PMAX(dataframe,period=self.buy_pmax_period.value)
        dataframe["buy_pmax"]=buy_pmax[f'pmX_{self.buy_pmax_period.value}_3_12_1']

        dataframe["buy_ssl_down"],dataframe["buy_ssl_up"] = SSLChannels(dataframe,length=self.buy_ssl_length.value)

        dataframe["buy_vpci"] = vpci(dataframe,period_short=self.buy_vpci_length.value,period_long=self.buy_vpci_length.value*4)

        info["buy_vfi"],info["buy_vfima"],info["buy_vfi_hist"] = vfi(info,length=(int)(self.buy_vfi_length.value/2),coef=0.2,signalLength=5,smoothVFI=False)

        buy_under_ma_df:pd.DataFrame=dataframe.tail(self.buy_under_ma_period.value)
        dataframe["buy_under_ma"]=buy_under_ma_df.max()["high"]

        buy_dataframe_long = resample_to_interval(dataframe, 30*self.buy_cdlhammer_length.value)
        buy_dataframe_long['cdlhammer'] = ta.CDLHAMMER(buy_dataframe_long)
        cdlhammer = resampled_merge(dataframe, buy_dataframe_long, fill_na=True)
        dataframe["cdlhammer"]=cdlhammer[f'resample_{30*self.buy_cdlhammer_length.value}_cdlhammer']
        buy_dataframe_long = resample_to_interval(dataframe, 30*self.buy_cdlinvertedhammer_length.value)
        buy_dataframe_long['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(buy_dataframe_long)
        cdlinvertedhammer = resampled_merge(dataframe, buy_dataframe_long, fill_na=True)
        dataframe["cdlinvertedhammer"]=cdlinvertedhammer[f'resample_{30*self.buy_cdlinvertedhammer_length.value}_cdlinvertedhammer']
        # buy_dataframe_long = resample_to_interval(dataframe, 60*self.buy_cdldragonflydoji_length.value)
        # buy_dataframe_long['cdldragonflydoji'] = ta.CDLDRAGONFLYDOJI(buy_dataframe_long)
        # cdldragonflydoji = resampled_merge(dataframe, buy_dataframe_long, fill_na=True)
        # dataframe["cdldragonflydoji"]=cdldragonflydoji[f'resample_{60*self.buy_cdldragonflydoji_length.value}_cdldragonflydoji']
        # buy_dataframe_long = resample_to_interval(dataframe, 60*self.buy_cdlpiercing_length.value)
        # buy_dataframe_long['cdlpiercing'] = ta.CDLPIERCING(buy_dataframe_long)
        # cdlpiercing = resampled_merge(dataframe, buy_dataframe_long, fill_na=True)
        # dataframe["cdlpiercing"]=cdlpiercing[f'resample_{60*self.buy_cdlpiercing_length.value}_cdlpiercing']
        # buy_dataframe_long = resample_to_interval(dataframe, 60*self.buy_cdlmorningstar_length.value)
        # buy_dataframe_long['cdlmorningstar'] = ta.CDLMORNINGSTAR(buy_dataframe_long)
        # cdlmorningstar = resampled_merge(dataframe, buy_dataframe_long, fill_na=True)
        # dataframe["cdlmorningstar"]=cdlmorningstar[f'resample_{60*self.buy_cdlmorningstar_length.value}_cdlmorningstar']
        # buy_dataframe_long = resample_to_interval(dataframe, 60*self.buy_cdl3whitesoldiers_length.value)
        # buy_dataframe_long['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(buy_dataframe_long)
        # cdl3whitesoldiers = resampled_merge(dataframe, buy_dataframe_long, fill_na=True)
        # dataframe["cdl3whitesoldiers"]=cdl3whitesoldiers[f'resample_{60*self.buy_cdl3whitesoldiers_length.value}_cdl3whitesoldiers']
        
        # buy_dataframe_long = resample_to_interval(dataframe, 60)
        # buy_dataframe_long['cdlhammer'] = ta.CDLHAMMER(buy_dataframe_long)
        # buy_dataframe_long['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(buy_dataframe_long)
        # buy_dataframe_long['cdldragonflydoji'] = ta.CDLDRAGONFLYDOJI(buy_dataframe_long)
        # buy_dataframe_long['cdlpiercing'] = ta.CDLPIERCING(buy_dataframe_long)
        # buy_dataframe_long['cdlmorningstar'] = ta.CDLMORNINGSTAR(buy_dataframe_long)
        # buy_dataframe_long['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(buy_dataframe_long)
        # buy_dataframe_long = resampled_merge(dataframe, buy_dataframe_long, fill_na=True)
        # dataframe["cdlhammer"]=buy_dataframe_long[f'resample_60_cdlhammer']
        # dataframe["cdlinvertedhammer"]=buy_dataframe_long[f'resample_60_cdlinvertedhammer']
        # dataframe["cdldragonflydoji"]=buy_dataframe_long[f'resample_60_cdldragonflydoji']
        # dataframe["cdlpiercing"]=buy_dataframe_long[f'resample_60_cdlpiercing']
        # dataframe["cdlmorningstar"]=buy_dataframe_long[f'resample_60_cdlmorningstar']
        # dataframe["cdl3whitesoldiers"]=buy_dataframe_long[f'resample_60_cdl3whitesoldiers']

        # dataframe['cdlhammer'] = ta.CDLHAMMER(dataframe)
        # dataframe['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(dataframe)
        # dataframe['cdldragonflydoji'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # dataframe['cdlpiercing'] = ta.CDLPIERCING(dataframe)
        # dataframe['cdlmorningstar'] = ta.CDLMORNINGSTAR(dataframe)
        # dataframe['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(dataframe)

        # # MACD - Moving Average Convergence Divergence
        # macd = ta.MACD(info)
        # info['macd'] = macd['macd']  # MACD - Blue TradingView Line (Bullish if on top)
        # info['macdsignal'] = macd['macdsignal']  # Signal - Orange TradingView Line (Bearish if on top)

        # # MFI - Money Flow Index (Under bought / Over sold & Over bought / Under sold / volume Indicator)
        # info['mfi'] = ta.MFI(info)

        # # Overlap Studies
        # # ---------------

        # # Bollinger Bands
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(info), window=20, stds=2)
        # info['bb_middleband'] = bollinger['mid']

        # info['sma9'] = ta.SMA(info, timeperiod=9)
        # info['sma50'] = ta.SMA(info, timeperiod=50)
        # info['sma200'] = ta.SMA(info, timeperiod=200)
        # info['sma18'] = ta.SMA(info, timeperiod=18)
        # info['sma100'] = ta.SMA(info, timeperiod=100)
        # info['sma400'] = ta.SMA(info, timeperiod=400)

        # # TEMA - Triple Exponential Moving Average
        # info['tema'] = ta.TEMA(info, timeperiod=9)



        # buy_boll = ta.BBANDS(info, timeperiod=self.buy_boll_period_1h.value,
        #                     nbdevup=float(self.buy_boll_stds_1h.value), nbdevdn=float(self.buy_boll_stds_1h.value), matype=self.buy_boll_matype_1h.value)
        # info["buy_boll_lower"] = buy_boll["lowerband"]

        # info["buy_rsi"] = ta.RSI(info, timeperiod=self.buy_rsi_period_1h.value)

        # buy_min_tail = info["close"].tail(self.buy_min_period_1h.value)
        # info["buy_min"] = buy_min_tail.min()

        # info["buy_ma1"] = ta.MA(info,timeperiod=self.buy_ma1_period_1h.value,matype=self.buy_ma1_type_1h.value)

        # info['buy_rolling_vwap'] = qtpylib.rolling_vwap(info,window=self.buy_roll_period_1h.value)

        # buy_stoch = qtpylib.stoch(info,window=self.buy_stoch_period_1h.value)
        # info['buy_slow_k']=buy_stoch["slow_k"]

        # info['buy_sar'] = ta.SAR(info,timeperiod=self.buy_sar_period_1h.value)

        # buy_macdfix=ta.MACDFIX(info,signalperiod=self.buy_macdfix_period_1h.value)
        # info["buy_macdfix_macd"]=buy_macdfix["macd"]
        # info["buy_macdfix_signal"]=buy_macdfix["macdsignal"]
        # info["buy_vidya"] = VIDYA(info,length=self.buy_vidya_length_1h.value)

        # info["buy_tke"],tke = TKE(info,length=self.buy_tke_length_1h.value)

        # info["buy_cci"] = ta.CCI(info,timeperiod=self.buy_cci_period_1h.value)

        
        # buy_pmax = PMAX(info,period=self.buy_pmax_period_1h.value)
        # info["buy_pmax"]=buy_pmax[f'pmX_{self.buy_pmax_period_1h.value}_3_12_1']

        # info["buy_ssl_down"],info["buy_ssl_up"] = SSLChannels(info,length=self.buy_ssl_length_1h.value)

        # info["buy_vpci"] = vpci(info,period_short=self.buy_vpci_length_1h.value,period_long=self.buy_vpci_length_1h.value*4)

        # info["buy_vfi"],info["buy_vfima"],info["buy_vfi_hist"] = vfi(info,length=self.buy_vfi_length_1h.value)

        # # MACD - Moving Average Convergence Divergence
        # macd = ta.MACD(info_2)
        # info_2['macd'] = macd['macd']  # MACD - Blue TradingView Line (Bullish if on top)
        # info_2['macdsignal'] = macd['macdsignal']  # Signal - Orange TradingView Line (Bearish if on top)

        # # MFI - Money Flow Index (Under bought / Over sold & Over bought / Under sold / volume Indicator)
        # info_2['mfi'] = ta.MFI(info_2)

        # # Overlap Studies
        # # ---------------

        # # Bollinger Bands
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(info_2), window=20, stds=2)
        # info_2['bb_middleband'] = bollinger['mid']

        # info_2['sma9'] = ta.SMA(info_2, timeperiod=9)
        # info_2['sma50'] = ta.SMA(info_2, timeperiod=50)
        # info_2['sma200'] = ta.SMA(info_2, timeperiod=200)
        # info_2['sma18'] = ta.SMA(info_2, timeperiod=18)
        # info_2['sma100'] = ta.SMA(info_2, timeperiod=100)
        # info_2['sma400'] = ta.SMA(info_2, timeperiod=400)

        # # TEMA - Triple Exponential Moving Average
        # info_2['tema'] = ta.TEMA(info_2, timeperiod=9)



        # buy_boll = ta.BBANDS(info_2, timeperiod=self.buy_boll_period_30m.value,
        #                     nbdevup=float(self.buy_boll_stds_30m.value), nbdevdn=float(self.buy_boll_stds_30m.value), matype=self.buy_boll_matype_30m.value)
        # info_2["buy_boll_lower"] = buy_boll["lowerband"]

        # info_2["buy_rsi"] = ta.RSI(info_2, timeperiod=self.buy_rsi_period_30m.value)

        # buy_min_tail = info_2["close"].tail(self.buy_min_period_30m.value)
        # info_2["buy_min"] = buy_min_tail.min()

        # info_2["buy_ma1"] = ta.MA(info_2,timeperiod=self.buy_ma1_period_30m.value,matype=self.buy_ma1_type_30m.value)

        # info_2['buy_rolling_vwap'] = qtpylib.rolling_vwap(info_2,window=self.buy_roll_period_30m.value)

        # buy_stoch = qtpylib.stoch(info_2,window=self.buy_stoch_period_30m.value)
        # info_2['buy_slow_k']=buy_stoch["slow_k"]

        # info_2['buy_sar'] = ta.SAR(info_2,timeperiod=self.buy_sar_period_30m.value)

        # buy_macdfix=ta.MACDFIX(info_2,signalperiod=self.buy_macdfix_period_30m.value)
        # info_2["buy_macdfix_macd"]=buy_macdfix["macd"]
        # info_2["buy_macdfix_signal"]=buy_macdfix["macdsignal"]
        # info_2["buy_vidya"] = VIDYA(info_2,length=self.buy_vidya_length_30m.value)

        # info_2["buy_tke"],tke = TKE(info_2,length=self.buy_tke_length_30m.value)

        # info_2["buy_cci"] = ta.CCI(info_2,timeperiod=self.buy_cci_period_30m.value)

        
        # buy_pmax = PMAX(info_2,period=self.buy_pmax_period_30m.value)
        # info_2["buy_pmax"]=buy_pmax[f'pmX_{self.buy_pmax_period_30m.value}_3_12_1']

        # info_2["buy_ssl_down"],info_2["buy_ssl_up"] = SSLChannels(info_2,length=self.buy_ssl_length_30m.value)

        # info_2["buy_vpci"] = vpci(info_2,period_short=self.buy_vpci_length_30m.value,period_long=self.buy_vpci_length_30m.value*4)

        # info_2["buy_vfi"],info_2["buy_vfima"],info_2["buy_vfi_hist"] = vfi(info_2,length=self.buy_vfi_length_30m.value)
        """
        add_sell_signal
        """

        sell_boll = ta.BBANDS(dataframe, timeperiod=self.sell_boll_period.value,
                            nbdevup=float(self.sell_boll_stds.value), nbdevdn=float(self.sell_boll_stds.value), matype=self.sell_boll_matype.value)
        dataframe["sell_boll_upper"] = sell_boll["upperband"]

        dataframe["sell_rsi"] = ta.RSI(dataframe, timeperiod=self.sell_rsi_period.value)

        sell_max_tail = dataframe["close"].tail(self.sell_max_period.value)
        dataframe["sell_max"] = sell_max_tail.max()

        dataframe['sell_rolling_vwap'] = qtpylib.rolling_vwap(dataframe,self.sell_roll_period.value)

        sell_stoch = qtpylib.stoch(dataframe,self.sell_stoch_period.value)
        dataframe["sell_slow_k"]=sell_stoch["slow_k"]

        dataframe['sell_sar'] = ta.SAR(dataframe,timeperiod=self.sell_sar_period.value)

        dataframe["sell_ma1"] = ta.MA(dataframe,timeperiod=self.sell_ma1_period.value,matype=self.sell_ma1_type.value)
        dataframe["sell_ma2"] = ta.MA(dataframe,timeperiod=self.sell_ma2_period.value,matype=self.sell_ma2_type.value)
        dataframe["sell_ma3"] = ta.MA(dataframe,timeperiod=self.sell_ma3_period.value,matype=self.sell_ma3_type.value)

        sell_macdfix=ta.MACDFIX(dataframe,signalperiod=self.sell_macdfix_period.value)
        dataframe["sell_macdfix_macd"]=sell_macdfix["macd"]
        dataframe["sell_macdfix_signal"]=sell_macdfix["macdsignal"]


        dataframe["sell_vidya"] = VIDYA(dataframe,length=self.sell_vidya_length.value)

        dataframe["sell_tke"],tke = TKE(dataframe,length=self.sell_tke_length.value)

        dataframe["sell_cci"] = ta.CCI(dataframe,timeperiod=self.sell_cci_period.value)

        sell_pmax = PMAX(dataframe,period=self.sell_pmax_period.value)
        dataframe["sell_pmax"]=sell_pmax[f'pmX_{self.sell_pmax_period.value}_3_12_1']

        dataframe["sell_ssl_down"],dataframe["sell_ssl_up"] = SSLChannels(dataframe,length=self.sell_ssl_length.value)

        dataframe["sell_vpci"] = vpci(dataframe,period_short=self.sell_vpci_length.value,period_long=self.sell_vpci_length.value*4)

        info["sell_vfi"],info["sell_vfima"],info["sell_vfi_hist"] = vfi(info,length=(int)(self.sell_vfi_length.value/2))

        sell_upper_ma_df:pd.DataFrame=dataframe.tail(self.sell_upper_ma_period.value)
        dataframe["sell_upper_ma"]=sell_upper_ma_df.min()["low"]

        sell_dataframe_long = resample_to_interval(dataframe, 30*self.sell_cdlhangingman_length.value)
        sell_dataframe_long['cdlhangingman'] = ta.CDLHANGINGMAN(sell_dataframe_long)
        cdlhangingman = resampled_merge(dataframe, sell_dataframe_long, fill_na=True)
        dataframe["cdlhangingman"]=cdlhangingman[f'resample_{30*self.sell_cdlhangingman_length.value}_cdlhangingman']
        sell_dataframe_long = resample_to_interval(dataframe, 30*self.sell_cdlshootingstar_length.value)
        sell_dataframe_long['cdlshootingstar'] = ta.CDLSHOOTINGSTAR(sell_dataframe_long)
        cdlshootingstar = resampled_merge(dataframe, sell_dataframe_long, fill_na=True)
        dataframe["cdlshootingstar"]=cdlshootingstar[f'resample_{30*self.sell_cdlshootingstar_length.value}_cdlshootingstar']
        # sell_dataframe_long = resample_to_interval(dataframe, 30*self.sell_cdlgravestonedoji_length.value)
        # sell_dataframe_long['cdlgravestonedoji'] = ta.CDLGRAVESTONEDOJI(sell_dataframe_long)
        # cdlgravestonedoji = resampled_merge(dataframe, sell_dataframe_long, fill_na=True)
        # dataframe["cdlgravestonedoji"]=cdlgravestonedoji[f'resample_{30*self.sell_cdlgravestonedoji_length.value}_cdlgravestonedoji']
        # sell_dataframe_long = resample_to_interval(dataframe, 30*self.sell_cdldarkcloudcover_length.value)
        # sell_dataframe_long['cdldarkcloudcover'] = ta.CDLDARKCLOUDCOVER(sell_dataframe_long)
        # cdldarkcloudcover = resampled_merge(dataframe, sell_dataframe_long, fill_na=True)
        # dataframe["cdldarkcloudcover"]=cdldarkcloudcover[f'resample_{30*self.sell_cdldarkcloudcover_length.value}_cdldarkcloudcover']
        # sell_dataframe_long = resample_to_interval(dataframe, 30*self.sell_cdleveningdojistar_length.value)
        # sell_dataframe_long['cdleveningdojistar'] = ta.CDLEVENINGDOJISTAR(sell_dataframe_long)
        # cdleveningdojistar = resampled_merge(dataframe, sell_dataframe_long, fill_na=True)
        # dataframe["cdleveningdojistar"]=cdleveningdojistar[f'resample_{30*self.sell_cdleveningdojistar_length.value}_cdleveningdojistar']
        # sell_dataframe_long = resample_to_interval(dataframe, 30*self.sell_cdleveningstar_length.value)
        # sell_dataframe_long['cdleveningstar'] = ta.CDLEVENINGSTAR(sell_dataframe_long)
        # cdleveningstar = resampled_merge(dataframe, sell_dataframe_long, fill_na=True)
        # dataframe["cdleveningstar"]=cdleveningstar[f'resample_{30*self.sell_cdleveningstar_length.value}_cdleveningstar']

        # sell_dataframe_long = resample_to_interval(dataframe, 60)
        # sell_dataframe_long['cdlhangingman'] = ta.CDLHANGINGMAN(sell_dataframe_long)
        # sell_dataframe_long['cdlshootingstar'] = ta.CDLSHOOTINGSTAR(sell_dataframe_long)
        # sell_dataframe_long['cdlgravestonedoji'] = ta.CDLGRAVESTONEDOJI(sell_dataframe_long)
        # sell_dataframe_long['cdldarkcloudcover'] = ta.CDLDARKCLOUDCOVER(sell_dataframe_long)
        # sell_dataframe_long['cdleveningdojistar'] = ta.CDLEVENINGDOJISTAR(sell_dataframe_long)
        # sell_dataframe_long['cdleveningstar'] = ta.CDLEVENINGSTAR(sell_dataframe_long)
        # sell_dataframe_long = resampled_merge(dataframe, sell_dataframe_long, fill_na=True)
        # dataframe["cdlhangingman"]=sell_dataframe_long[f'resample_60_cdlhangingman']
        # dataframe["cdlshootingstar"]=sell_dataframe_long[f'resample_60_cdlshootingstar']
        # dataframe["cdlgravestonedoji"]=sell_dataframe_long[f'resample_60_cdlgravestonedoji']
        # dataframe["cdldarkcloudcover"]=sell_dataframe_long[f'resample_60_cdldarkcloudcover']
        # dataframe["cdleveningdojistar"]=sell_dataframe_long[f'resample_60_cdleveningdojistar']
        # dataframe["cdleveningstar"]=sell_dataframe_long[f'resample_60_cdleveningstar']

        # dataframe['cdlhangingman'] = ta.CDLHANGINGMAN(dataframe)
        # dataframe['cdlshootingstar'] = ta.CDLSHOOTINGSTAR(dataframe)
        # dataframe['cdlgravestonedoji'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # dataframe['cdldarkcloudcover'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # dataframe['cdleveningdojistar'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # dataframe['cdleveningstar'] = ta.CDLEVENINGSTAR(dataframe)


        # info["sell_macdfix_signal"]=sell_macdfix["macdsignal"]

        # sell_boll = ta.BBANDS(info, timeperiod=self.sell_boll_period_1h.value,
        #                     nbdevup=float(self.sell_boll_stds_1h.value), nbdevdn=float(self.sell_boll_stds_1h.value), matype=self.sell_boll_matype_1h.value)
        # info["sell_boll_upper"] = sell_boll["upperband"]

        # info["sell_rsi"] = ta.RSI(info, timeperiod=self.sell_rsi_period_1h.value)

        # sell_max_tail = info["close"].tail(self.sell_max_period_1h.value)
        # info["sell_max"] = sell_max_tail.max()

        # info['sell_rolling_vwap'] = qtpylib.rolling_vwap(info,self.sell_roll_period_1h.value)

        # sell_stoch = qtpylib.stoch(info,self.sell_stoch_period_1h.value)
        # info["sell_slow_k"]=sell_stoch["slow_k"]

        # info['sell_sar'] = ta.SAR(info,timeperiod=self.sell_sar_period_1h.value)

        # info["sell_ma1"] = ta.MA(info,timeperiod=self.sell_ma1_period_1h.value,matype=self.sell_ma1_type_1h.value)

        # sell_macdfix=ta.MACDFIX(info,signalperiod=self.sell_macdfix_period_1h.value)
        # info["sell_macdfix_macd"]=sell_macdfix["macd"]

        # info["sell_vidya"] = VIDYA(info,length=self.sell_vidya_length_1h.value)

        # info["sell_tke"],tke = TKE(info,length=self.sell_tke_length_1h.value)

        # info["sell_cci"] = ta.CCI(info,timeperiod=self.sell_cci_period_1h.value)

        # sell_pmax = PMAX(info,period=self.sell_pmax_period_1h.value)
        # info["sell_pmax"]=sell_pmax[f'pmX_{self.sell_pmax_period_1h.value}_3_12_1']

        # info["sell_ssl_down"],info["sell_ssl_up"] = SSLChannels(info,length=self.sell_ssl_length_1h.value)

        # info["sell_vpci"] = vpci(info,period_short=self.sell_vpci_length_1h.value,period_long=self.sell_vpci_length_1h.value*4)

        # info["sell_vfi"],info["sell_vfima"],info["sell_vfi_hist"] = vfi(info,length=self.sell_vfi_length_1h.value)


        # info_2["sell_macdfix_signal"]=sell_macdfix["macdsignal"]

        # sell_boll = ta.BBANDS(info_2, timeperiod=self.sell_boll_period_30m.value,
        #                     nbdevup=float(self.sell_boll_stds_30m.value), nbdevdn=float(self.sell_boll_stds_30m.value), matype=self.sell_boll_matype_30m.value)
        # info_2["sell_boll_upper"] = sell_boll["upperband"]

        # info_2["sell_rsi"] = ta.RSI(info_2, timeperiod=self.sell_rsi_period_30m.value)

        # sell_max_tail = info_2["close"].tail(self.sell_max_period_30m.value)
        # info_2["sell_max"] = sell_max_tail.max()

        # info_2['sell_rolling_vwap'] = qtpylib.rolling_vwap(info_2,self.sell_roll_period_30m.value)

        # sell_stoch = qtpylib.stoch(info_2,self.sell_stoch_period_30m.value)
        # info_2["sell_slow_k"]=sell_stoch["slow_k"]

        # info_2['sell_sar'] = ta.SAR(info_2,timeperiod=self.sell_sar_period_30m.value)

        # info_2["sell_ma1"] = ta.MA(info_2,timeperiod=self.sell_ma1_period_30m.value,matype=self.sell_ma1_type_30m.value)

        # sell_macdfix=ta.MACDFIX(info_2,signalperiod=self.sell_macdfix_period_30m.value)
        # info_2["sell_macdfix_macd"]=sell_macdfix["macd"]

        # info_2["sell_vidya"] = VIDYA(info_2,length=self.sell_vidya_length_30m.value)

        # info_2["sell_tke"],tke = TKE(info_2,length=self.sell_tke_length_30m.value)

        # info_2["sell_cci"] = ta.CCI(info_2,timeperiod=self.sell_cci_period_30m.value)

        # sell_pmax = PMAX(info_2,period=self.sell_pmax_period_30m.value)
        # info_2["sell_pmax"]=sell_pmax[f'pmX_{self.sell_pmax_period_30m.value}_3_12_1']

        # info_2["sell_ssl_down"],info_2["sell_ssl_up"] = SSLChannels(info_2,length=self.sell_ssl_length_30m.value)

        # info_2["sell_vpci"] = vpci(info_2,period_short=self.sell_vpci_length_30m.value,period_long=self.sell_vpci_length_30m.value*4)

        # info_2["sell_vfi"],info_2["sell_vfima"],info_2["sell_vfi_hist"] = vfi(info_2,length=self.sell_vfi_length_30m.value)

        """
        add_share_signals
        """

        dataframe=merge_informative_pair(dataframe=dataframe,timeframe=self.timeframe,timeframe_inf=info_time,informative=info.copy())
        # dataframe=merge_informative_pair(dataframe=dataframe,timeframe=self.timeframe,timeframe_inf=info_time_2,informative=info_2.copy())
        # print(dataframe.tail(1))
        fxgt_list=[            
            'ADA/USDT',
            'XRP/USDT',
            'BCH/USDT',
            'LTC/USDT',
            'DOT/USDT',
            'XLM/USDT',
            'SHIB/USDT',
            'DOGE/USDT',
            'LINK/USDT',
            'LUNA/USDT',
            'ENJ/USDT',
            'MANA/USDT',
            'CHZ/USDT',
            'DGB/USDT',
            'ZRX/USDT',
            'REN/USDT',
            'UNI/USDT',
            'SNX/USDT',
            'MATIC/USDT',
            'COMP/USDT',
            'AAVE/USDT'
            ]
        if metadata["pair"] in fxgt_list:
            analyzed_df,analyzed_date=self.dp.get_analyzed_dataframe(metadata['pair'], self.timeframe)  
            if not analyzed_df.empty:
                temp=analyzed_df.iloc[-1]
                jst_date=analyzed_date.astimezone(datetime.timezone(datetime.timedelta(hours=+9)))
                buy_bool=not np.isnan(analyzed_df.at[analyzed_df.index[-1],"buy"])
                sell_bool=not np.isnan(analyzed_df.at[analyzed_df.index[-1],"sell"])
                # print(f'{pair}') 
                self.mgm_logger('custom', "fxgt",f'{metadata["pair"]:<11}BUY:{str(buy_bool):<6} SELL:{str(sell_bool):<6} {jst_date}')
                # print(f'{metadata["pair"]:<11}BUY:{str(buy_bool):<6} SELL:{str(sell_bool):<6} {jst_date}') 
            else:
                print("Nothing analyzed_df")
        return dataframe
        

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the buy trend with weighted buy signals used in MoniGoMani's DataFrame per pair.

        :param dataframe: (DataFrame) DataFrame with data from the exchange and all mandatory indicator data populated
        :param metadata: (dict) Additional information, like the currently traded pair
        :return DataFrame: DataFrame for MoniGoMani with all mandatory weighted buy signals populated
        """

        # Keep this call to populate the conditions responsible for the weights of your buy signals
        dataframe = self._populate_trend('buy', dataframe, metadata)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the buy trend with weighted buy signals used in MoniGoMani's DataFrame per pair.

        :param dataframe: (DataFrame) DataFrame with data from the exchange,
            all mandatory indicator and weighted buy signal data populated
        :param metadata: (dict) Additional information, like the currently traded pair
        :return DataFrame: DataFrame for MoniGoMani with all mandatory weighted sell signals populated
        """

        # Keep this call to populate the conditions responsible for the weights of your sell signals
        dataframe = self._populate_trend('sell', dataframe, metadata)

        return dataframe
