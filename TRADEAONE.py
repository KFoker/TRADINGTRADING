import MetaTrader5 as mt5  # type: ignore

import pandas as pd

import numpy as np

import time

import logging

from datetime import datetime, timedelta

import sys

import talib  # type: ignore

from collections import deque

import math

import threading

from typing import Dict, List, Tuple, Optional, Any

import traceback

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import os
import json
from collections import defaultdict

# 配置专业日志系统

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

    datefmt='%H:%M:%S',

    handlers=[

        logging.FileHandler("professional_complex_fixed.log", encoding='utf-8'),

        logging.StreamHandler(sys.stdout)

    ]

)

logger = logging.getLogger("ProfessionalComplexFixed")


class ProfessionalComplexConfig:
    """专业复杂配置 - 保持所有复杂性但修复数据源问题"""

    # 账户信息

    LOGIN = 70724849

    PASSWORD = "-mY4NkKc"

    SERVER = "PlotioGlobalFinancial-Demo"

    # 品种配置 - 只使用Gold

    SYMBOL_CANDIDATES = [

        "Gold",  # 原符号

        "GOLD",  # 大写

        "XAUUSD",  # 标准符号

        "Gold Spot",  # 完整名称

    ]

    DEFAULT_SYMBOL = "Gold"

    # 品种规格

    POINT_VALUE = 1.0  # 黄金每点约1美元

    TICK_SIZE = 0.01

    POINT = 0.01

    # 仓位管理参数 - 保持复杂性

    MIN_LOT = 0.1

    MAX_LOT = 5.0

    LOT_STEP = 0.1

    RISK_PER_TRADE = 0.002

    MAX_DAILY_TRADES = 50

    MAX_CONCURRENT_TRADES = 3

    MAX_DRAWDOWN = 0.05

    # 盈亏比配置
    MIN_RISK_REWARD_RATIO = 1.2  # 最小盈亏比要求（1.2:1），低于此值拒绝开仓

    # 交易成本配置
    COMMISSION_PER_LOT = 0.0  # 每手手续费（美元），需要根据实际经纪商设置
    SPREAD_COST_ENABLED = True  # 是否考虑点差成本
    SPREAD_COST_MULTIPLIER = 0.3  # 点差成本倍数（0.0-1.0），用于调整点差影响程度，0.3表示只考虑30%的点差影响（降低以增加开仓机会）
    # 盈亏比对手数的影响：盈亏比越低，手数减少越多
    RR_POSITION_ADJUSTMENT = True  # 是否根据盈亏比调整仓位
    MIN_RR_FOR_FULL_SIZE = 2.5  # 盈亏比达到此值时才使用满仓

    # 趋势启动检测参数
    TREND_START_DETECTION = {
        'ENABLE': True,  # 启用趋势启动检测
        'ADX_RISING_THRESHOLD': 18,  # ADX上升阈值
        'ADX_PREV_THRESHOLD': 15,  # 之前ADX阈值
        'MOMENTUM_ACCELERATION': 1.5,  # 动量加速倍数
        'MIN_SIGNAL_STRENGTH': 0.35  # 最小信号强度
    }

    # 趋势衰竭检测参数
    TREND_EXHAUSTION = {
        'ENABLE': True,
        'MIN_PROFIT_PCT': 0.005,  # 至少0.5%盈利才检测
        'SIGNALS_REQUIRED': 2  # 需要至少2个信号
    }

    # 动态止盈参数
    DYNAMIC_TAKE_PROFIT = {
        'ENABLE': True,
        'MIN_PROFIT_PCT': 0.005,  # 至少0.5%盈利才启用
        'UPDATE_INTERVAL': 10,  # 每10秒更新一次
        'STRONG_TREND_ATR_MULT': 3.0,  # 强趋势ATR倍数
        'MEDIUM_TREND_ATR_MULT': 2.5,  # 中等趋势ATR倍数
        'WEAK_TREND_ATR_MULT': 2.0,  # 弱趋势ATR倍数
        'MIN_ADX_FOR_DYNAMIC': 25,  # 最小ADX值才使用动态止盈（单边明确趋势）
        'USE_FOR_STRONG_TREND_ONLY': True  # 只在强趋势时使用动态止盈
    }

    # 多目标止盈参数
    MULTI_TARGET_TP = {
        'ENABLE': True,
        'TARGET_COUNT': 2,  # 只分两次止盈
        'FIRST_TARGET_PCT': 0.8,  # 第一次止盈80%仓位
        'SECOND_TARGET_PCT': 0.2,  # 第二次止盈20%仓位
        'FIRST_TARGET_RR': 0.8,  # 第一段盈亏比100%（1:1）
        'SECOND_TARGET_RR': 1.0,  # 第二段盈亏比120%（1.2:1）
        'USE_FOR_RANGING': True,  # 震荡市使用多目标止盈
        'USE_FOR_WEAK_TREND': True  # 弱趋势使用多目标止盈
    }

    # 盈利回撤控制参数
    PROFIT_DRAWDOWN_CONTROL = {
        'ENABLE': True,
        'MIN_PEAK_PROFIT_USD': 8.0,  # 最小峰值盈利（美元）才启用保护：盈利价与入场价达8美元以上
        'MAX_DRAWDOWN_USD': 3.0  # 最大盈利回撤（美元）：回撤3美元主动止盈
    }

    # Tick处理参数 - 优化

    TICK_BUFFER_SIZE = 500

    PRICE_BUFFER_SIZE = 200

    PROCESSING_INTERVAL = 0.05

    MIN_TICKS_FOR_ANALYSIS = 30  # 降低要求

    # 多时间框架分析 - 保持复杂性

    TICK_TIMEFRAMES = {

        'ULTRA_SHORT': 10,

        'SHORT': 30,

        'MEDIUM': 100,

        'LONG': 200

    }

    # 技术指标参数 - 完整复杂设置

    TECHNICAL_INDICATORS = {

        'RSI': {

            'PERIODS': [3, 6, 14],

            'OVERSOLD': [25, 30, 35],

            'OVERBOUGHT': [75, 70, 65]

        },

        'MACD': {

            'FAST': 12,

            'SLOW': 26,

            'SIGNAL': 9

        },

        'STOCHASTIC': {

            'K_PERIOD': 14,

            'D_PERIOD': 3,

            'SLOWING': 3

        },

        'EMA': {

            'PERIODS': [5, 15, 30, 60]

        },

        'BOLLINGER': {

            'PERIOD': 20,

            'STD_DEV': 2.0,

            'BANDS': [1.0, 1.5, 2.0, 2.5]

        },

        'ATR': {

            'PERIOD': 14

        },

        'ADX': {

            'PERIOD': 14

        },

        'CCI': {

            'PERIOD': 20

        },

        'WILLIAMSR': {

            'PERIOD': 14

        },

        'KDJ': {

            'RSV_PERIOD': 9,  # RSV周期

            'K_PERIOD': 3,  # K值平滑周期

            'D_PERIOD': 3  # D值平滑周期

        }

    }

    # 市场状态识别参数 - 优化阈值

    MARKET_STATE_PARAMS = {

        'TRENDING': {

            'ADX_THRESHOLD': 20,  # 降低阈值

            'EMA_ALIGNMENT': 3,  # 需要3个EMA同向

            'PRICE_MOMENTUM': 0.0003

        },

        'RANGING': {

            'ATR_RATIO_MAX': 0.0004,

            'BB_WIDTH_RATIO': 0.002,

            'PRICE_OSCILLATION': 0.001

        },

        'VOLATILE': {

            'ATR_RATIO_MIN': 0.0006,

            'PRICE_SPIKE_FREQUENCY': 3,

            'VOLUME_SPIKE_RATIO': 2.0

        }

    }

    # 信号生成参数 - 保持复杂权重系统

    SIGNAL_GENERATION = {

        'CONFIRMATION_REQUIRED': 2,  # 降低要求

        'MIN_STRENGTH': 0.30,  # 进一步降低阈值以捕捉更多交易机会

        'WEIGHT_SYSTEM': {

            'TRENDING': {

                'TREND_INDICATORS': 0.35,

                'MOMENTUM_INDICATORS': 0.25,

                'VOLATILITY_INDICATORS': 0.20,

                'PATTERN_RECOGNITION': 0.10,

                'PRICE_ACTION': 0.10

            },

            'RANGING': {

                'OSCILLATORS': 0.30,

                'SUPPORT_RESISTANCE': 0.25,

                'VOLATILITY_INDICATORS': 0.20,

                'PRICE_PATTERNS': 0.15,

                'TREND_INDICATORS': 0.10

            },

            'VOLATILE': {

                'VOLATILITY_INDICATORS': 0.35,

                'PRICE_ACTION': 0.25,

                'BREAKOUT_SIGNALS': 0.20,

                'TREND_INDICATORS': 0.15,

                'MOMENTUM_INDICATORS': 0.05

            }

        },

        'FILTERS': {

            'SPREAD_MAX': 999999,  # 移除点差过滤（设置为极大值）

            'MIN_VOLATILITY': 0.00005,

            'MIN_TICKS_BETWEEN_SIGNALS': 3

        }

    }

    # 风险管理参数 - 专业设置

    RISK_MANAGEMENT = {

        'POSITION_SIZING': {

            'KELLY_FRACTION': 0.3,

            'VOLATILITY_ADJUSTMENT': True,

            'CORRELATION_FACTOR': 0.8

        },

        'STOP_LOSS': {

            'MULTI_LAYER': True,

            'LEVELS': [

                {'DISTANCE': 'ATR_1.0', 'SIZE_PERCENT': 0.4},

                {'DISTANCE': 'ATR_1.5', 'SIZE_PERCENT': 0.3},

                {'DISTANCE': 'ATR_2.0', 'SIZE_PERCENT': 0.3}

            ],

            'TRAILING': {

                'ACTIVATION_PERCENT': 0.002,

                'STEP_SIZE': 0.001

            }

        },

        'TAKE_PROFIT': {

            'MULTI_TARGET': True,

            'TARGETS': [

                {'PRICE_LEVEL': 'R1', 'CLOSE_PERCENT': 0.25},

                {'PRICE_LEVEL': 'R2', 'CLOSE_PERCENT': 0.35},

                {'PRICE_LEVEL': 'R3', 'CLOSE_PERCENT': 0.40}

            ],

            'DYNAMIC_ADJUSTMENT': True

        }

    }

    # 执行参数 - 优化

    EXECUTION_PARAMS = {

        'ORDER_TYPES': ['MARKET', 'LIMIT', 'STOP'],

        'MAX_SLIPPAGE': 30,

        'RETRY_COUNT': 3,

        'RETRY_DELAY': 0.5

    }


class DataSourceValidator:
    """数据源验证器 - 专门解决点差问题"""

    def __init__(self):

        self.valid_symbol = None

        self.symbol_info = None

        self.connection_quality = {

            'success_rate': 0.0,

            'avg_spread': 0.0,

            'tick_frequency': 0.0,

            'last_success': 0.0

        }

        self.test_results = {}

    @staticmethod
    def _get_tick_value(tick, field_name, default=0.0):

        """安全获取tick字段值，支持numpy结构化数组和普通对象"""

        try:

            # 尝试字典式访问（numpy结构化数组）

            if hasattr(tick, '__getitem__'):

                try:

                    return float(tick[field_name])

                except (KeyError, TypeError, IndexError):

                    pass

            # 尝试属性访问（普通对象）

            if hasattr(tick, field_name):
                return float(getattr(tick, field_name))

            return default

        except Exception:

            return default

    def find_valid_symbol(self) -> Optional[str]:

        """寻找有效的交易品种"""

        logger.info("🔍 开始寻找有效交易品种...")

        for symbol in ProfessionalComplexConfig.SYMBOL_CANDIDATES:

            if self._test_symbol_viability(symbol):
                self.valid_symbol = symbol

                logger.info(f"✅ 找到有效品种: {symbol}")

                return symbol

        # 尝试获取所有可用品种，优先Gold相关

        all_symbols = self._get_all_available_symbols()

        if all_symbols:

            # 优先查找包含Gold的品种

            gold_symbols = [s for s in all_symbols if 'GOLD' in s.upper() or 'XAU' in s.upper()]

            # 按优先级排序：Gold > GOLD > 其他

            gold_symbols.sort(key=lambda x: (

                0 if x.upper() == 'GOLD' else

                1 if 'GOLD' in x.upper() and 'XAU' not in x.upper() else

                2

            ))

            for symbol in gold_symbols[:10]:  # 测试前10个

                if self._test_symbol_viability(symbol):
                    self.valid_symbol = symbol

                    logger.info(f"✅ 找到有效品种: {symbol}")

                    return symbol

        logger.error("❌ 未找到有效交易品种")

        return None

    def _test_symbol_viability(self, symbol: str) -> bool:

        """测试品种可行性"""

        try:

            logger.info(f"🧪 测试品种: {symbol}")

            # 检查品种是否存在

            symbol_info = mt5.symbol_info(symbol)

            if not symbol_info:
                logger.warning(f"  品种不存在: {symbol}")

                return False

            # 检查交易权限

            if not symbol_info.visible:
                logger.warning(f"  品种不可见: {symbol}")

                return False

            # 选择品种

            if not mt5.symbol_select(symbol, True):
                logger.warning(f"  无法选择品种: {symbol}")

                return False

            # 测试数据质量

            return self._test_data_quality(symbol, symbol_info)

        except Exception as e:

            logger.error(f"测试品种异常: {symbol} - {str(e)}")

            return False

    def _test_data_quality(self, symbol: str, symbol_info: Any) -> bool:

        """测试数据质量"""

        try:

            # 首先尝试实时Tick验证（最快最可靠）
            logger.info(f"  先测试实时Tick数据...")
            if self._test_realtime_tick_quality(symbol, symbol_info):
                logger.info(f"  ✅ 实时Tick验证成功，品种可用")
                return True

            # 如果实时Tick不可用，尝试历史数据（扩大时间范围）
            logger.info(f"  实时Tick不可用，尝试历史数据...")
            end_time = datetime.now()

            # 尝试多个时间范围，从短到长
            time_ranges = [
                timedelta(minutes=5),
                timedelta(minutes=30),
                timedelta(hours=1),
                timedelta(hours=24)
            ]

            ticks = None
            used_time_range = None

            for time_range in time_ranges:
                start_time = end_time - time_range
                ticks = mt5.copy_ticks_range(symbol, start_time, end_time, mt5.COPY_TICKS_ALL)

                if ticks is not None:
                    ticks_len = ticks.size if hasattr(ticks, 'size') else len(ticks) if ticks else 0
                    if ticks_len >= 10:
                        used_time_range = time_range
                        logger.info(f"  ✅ 成功获取历史数据: {time_range}, {ticks_len}个Tick")
                        break
                    elif ticks_len > 0:
                        logger.debug(f"  时间范围{time_range}: 获取到{ticks_len}个Tick（不足10个，继续尝试）")

            # 如果所有时间范围都失败，尝试使用K线数据作为备选
            if ticks is None or (hasattr(ticks, 'size') and ticks.size == 0) or (
                    not hasattr(ticks, 'size') and len(ticks) == 0):
                logger.warning(f"  历史Tick数据不可用，尝试K线数据...")
                try:
                    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
                    if rates is not None and len(rates) > 0:
                        logger.info(f"  ✅ K线数据可用: {len(rates)}根K线，品种可用")
                        self.symbol_info = symbol_info
                        return True
                except Exception as e:
                    logger.debug(f"  K线数据获取失败: {str(e)}")

                logger.warning(f"  所有数据源都不可用")
                return False

            # 使用size属性检查numpy数组

            if hasattr(ticks, 'size'):

                if ticks.size == 0:
                    logger.warning(f"  数据不足: {ticks.size}个Tick")

                    return False

                ticks_len = ticks.size

            else:

                # 如果是列表或其他类型

                ticks_len = len(ticks) if ticks else 0

                if ticks_len == 0:
                    logger.warning(f"  数据不足: {ticks_len}个Tick")

                    return False

            # 分析Tick数据（移除点差过滤，只检查价格有效性）

            spreads = []

            valid_ticks = 0

            invalid_reasons = {'invalid_price': 0, 'total_checked': 0}

            for tick in ticks:

                invalid_reasons['total_checked'] += 1

                # 使用安全方法获取tick值

                ask = self._get_tick_value(tick, 'ask')

                bid = self._get_tick_value(tick, 'bid')

                # 只检查价格有效性，不检查点差

                if ask <= 0 or bid <= 0 or ask <= bid:
                    invalid_reasons['invalid_price'] += 1

                    continue

                # 计算点差用于统计（不用于过滤）

                spread_points = (ask - bid) * 10000

                spreads.append(spread_points)

                valid_ticks += 1

            # 输出详细调试信息

            if spreads:

                min_spread = min(spreads)

                max_spread = max(spreads)

                avg_spread = np.mean(spreads)

                median_spread = np.median(spreads)

                logger.info(f"  Tick分析: 总计{invalid_reasons['total_checked']}个, "

                            f"有效{valid_ticks}个, "

                            f"无效价格{invalid_reasons['invalid_price']}个")

                logger.info(f"  点差统计: 最小{min_spread:.1f}点, 最大{max_spread:.1f}点, "

                            f"平均{avg_spread:.1f}点, 中位数{median_spread:.1f}点")

            else:

                logger.info(f"  Tick分析: 总计{invalid_reasons['total_checked']}个, "

                            f"有效{valid_ticks}个, "

                            f"无效价格{invalid_reasons['invalid_price']}个")

            # 如果历史数据不足，但至少有数据，降低要求
            if valid_ticks < 5:
                logger.warning(f"  历史Tick数据不足: {valid_ticks}个有效Tick")
                # 如果已经尝试过实时Tick，不再重复尝试
                # 如果历史数据至少有一些，认为品种可用（可能是市场刚开市）
                if valid_ticks >= 1:
                    logger.info(f"  历史数据较少但可用，品种可能刚开市")
                    self.symbol_info = symbol_info
                    return True
                else:
                    logger.warning(f"  历史数据完全不可用")
                    return False

            avg_spread = np.mean(spreads) if spreads else 0

            max_spread = max(spreads) if spreads else 0

            logger.info(f"  数据质量: {valid_ticks}个有效Tick, 平均点差: {avg_spread:.1f}, 最大点差: {max_spread:.1f}")

            # 检查价格范围

            prices = []

            for tick in ticks:

                bid = self._get_tick_value(tick, 'bid')

                if bid > 0:
                    prices.append(bid)

            if prices:

                price_range = (max(prices) - min(prices)) / min(prices) if min(prices) > 0 else 0

                if price_range > 0.1:  # 价格变化过大

                    logger.warning(f"  价格变化异常: {price_range:.2%}")

                    return False

            self.symbol_info = symbol_info

            self.connection_quality['avg_spread'] = avg_spread

            # 安全计算成功率

            total_ticks = ticks.size if hasattr(ticks, 'size') else len(ticks)

            self.connection_quality['success_rate'] = valid_ticks / total_ticks if total_ticks > 0 else 0

            return True

        except Exception as e:

            logger.error(f"测试数据质量异常: {str(e)}")

            return False

    def _test_realtime_tick_quality(self, symbol: str, symbol_info: Any) -> bool:

        """使用实时Tick数据测试数据质量（移除点差过滤）"""

        try:

            logger.info(f"  测试实时Tick数据...")

            # 获取多个实时tick样本

            valid_samples = 0

            spreads = []

            # 增加尝试次数和等待时间，提高成功率
            max_attempts = 20  # 从10次增加到20次
            wait_interval = 0.2  # 从0.1秒增加到0.2秒，给MT5更多时间响应

            for i in range(max_attempts):

                tick = mt5.symbol_info_tick(symbol)

                if tick:

                    ask = self._get_tick_value(tick, 'ask')

                    bid = self._get_tick_value(tick, 'bid')

                    # 只检查价格有效性，不检查点差

                    if ask > bid > 0:

                        spread = (ask - bid) * 10000

                        spreads.append(spread)

                        valid_samples += 1

                        # 如果已经获取到足够的样本，提前退出
                        if valid_samples >= 5:
                            break

                time.sleep(wait_interval)  # 等待获取下一个tick

            if valid_samples >= 3:  # 至少需要3个有效样本

                avg_spread = np.mean(spreads) if spreads else 0

                logger.info(f"  实时Tick验证成功: {valid_samples}个有效样本, 平均点差: {avg_spread:.1f}点")

                self.symbol_info = symbol_info

                self.connection_quality['avg_spread'] = avg_spread

                self.connection_quality['success_rate'] = valid_samples / max_attempts

                return True

            elif valid_samples >= 1:
                # 如果至少有一个有效样本，也认为品种可用（可能是市场刚开市或数据较少）
                avg_spread = np.mean(spreads) if spreads else 0
                logger.info(f"  实时Tick验证: {valid_samples}个有效样本（较少，但品种可用）, 平均点差: {avg_spread:.1f}点")
                self.symbol_info = symbol_info
                self.connection_quality['avg_spread'] = avg_spread
                self.connection_quality['success_rate'] = valid_samples / max_attempts
                return True

            else:

                logger.warning(f"  实时Tick验证失败: 仅{valid_samples}个有效样本")

                return False

        except Exception as e:

            logger.error(f"测试实时Tick质量异常: {str(e)}")

            return False

    def _get_all_available_symbols(self) -> List[str]:

        """获取所有可用品种"""

        try:

            all_symbols = mt5.symbols_get()

            return [s.name for s in all_symbols] if all_symbols else []

        except:

            return []

    def get_symbol_info(self) -> Dict[str, Any]:

        """获取品种信息"""

        if not self.symbol_info:
            return {}

        return {

            'name': self.symbol_info.name,

            'bid': self.symbol_info.bid,

            'ask': self.symbol_info.ask,

            'spread': (self.symbol_info.ask - self.symbol_info.bid) * 10000,

            'point': self.symbol_info.point,

            'digits': self.symbol_info.digits,

            'trade_mode': self.symbol_info.trade_mode

        }


class ProfessionalTickDataEngine:
    """专业Tick数据引擎 - 保持复杂性但修复数据源"""

    def __init__(self, data_validator: DataSourceValidator):

        self.data_validator = data_validator

        self.symbol = data_validator.valid_symbol

        self.tick_buffer = deque(maxlen=ProfessionalComplexConfig.TICK_BUFFER_SIZE)

        self.price_buffer = deque(maxlen=ProfessionalComplexConfig.PRICE_BUFFER_SIZE)

        self.volume_buffer = deque(maxlen=100)

        self.high_buffer = deque(maxlen=200)

        self.low_buffer = deque(maxlen=200)

        self.indicators_cache = {}

        self.initialized = False

        self.data_quality = {

            'total_ticks': 0,

            'valid_ticks': 0,

            'avg_spread': 0.0,

            'tick_frequency': 0.0,

            'last_quality_check': 0.0

        }

        # 历史指标存储（用于趋势启动检测）
        self.previous_indicators = {}  # 存储上一次的指标值

    def process_tick_data(self) -> bool:

        """处理Tick数据 - 增强容错"""

        try:

            tick = mt5.symbol_info_tick(self.symbol)

            if not tick:
                # 静默返回False，避免日志过多
                return False

            # 深度数据验证

            if not self._validate_tick_quality(tick):
                # 静默返回False，避免日志过多
                return False

            # 创建增强的Tick记录

            tick_data = self._create_enhanced_tick_record(tick)

            # 更新所有缓冲区

            self._update_data_buffers(tick_data)

            # 更新数据质量指标

            self._update_data_quality_metrics(tick_data)

            # 检查初始化状态

            if not self.initialized and len(self.tick_buffer) >= ProfessionalComplexConfig.MIN_TICKS_FOR_ANALYSIS:
                self.initialized = True

                logger.info(f"✅ 数据引擎初始化完成 - 有效Tick: {self.data_quality['valid_ticks']}")

                self._report_initialization_status()

            return True

        except Exception as e:

            # 只在关键异常时记录，避免日志过多
            if self.data_quality['total_ticks'] % 100 == 0:  # 每100个tick记录一次异常
                logger.warning(f"处理Tick数据异常 (已处理{self.data_quality['total_ticks']}个): {str(e)}")

            return False

    def _validate_tick_quality(self, tick: Any) -> bool:

        """验证Tick数据质量（移除点差过滤）"""

        # 使用安全方法获取tick值

        bid = DataSourceValidator._get_tick_value(tick, 'bid')

        ask = DataSourceValidator._get_tick_value(tick, 'ask')

        # 只检查价格有效性，不检查点差

        if bid <= 0 or ask <= 0:
            return False

        if ask <= bid:
            return False

        # 价格变化合理性检查（防止异常价格跳变）

        if self.tick_buffer:

            last_tick = self.tick_buffer[-1]

            price_change = abs(bid - last_tick['bid']) / last_tick['bid'] if last_tick['bid'] > 0 else 0

            if price_change > 0.01:  # 1%以上的异常变化

                logger.warning(f"价格异常变化: {price_change:.2%}")

                return False

        return True

    def _create_enhanced_tick_record(self, tick: Any) -> Dict[str, Any]:

        """创建增强的Tick记录"""

        # 使用安全方法获取tick值

        bid = DataSourceValidator._get_tick_value(tick, 'bid')

        ask = DataSourceValidator._get_tick_value(tick, 'ask')

        last = DataSourceValidator._get_tick_value(tick, 'last', bid)

        volume = int(DataSourceValidator._get_tick_value(tick, 'volume', 0))

        spread = (ask - bid) * 10000

        mid_price = (bid + ask) / 2

        return {

            'timestamp': time.time(),

            'datetime': datetime.now(),

            'bid': float(bid),

            'ask': float(ask),

            'last': float(last),

            'volume': volume,

            'spread': spread,

            'mid_price': mid_price,

            'tick_direction': self._calculate_tick_direction(mid_price),

            'price_momentum': self._calculate_price_momentum(mid_price),

            'volume_profile': self._analyze_volume_profile()

        }

    def _calculate_tick_direction(self, current_price: float) -> int:

        """计算Tick方向"""

        if not self.price_buffer:
            return 0

        last_price = self.price_buffer[-1]

        if current_price > last_price:

            return 1

        elif current_price < last_price:

            return -1

        else:

            return 0

    def _calculate_price_momentum(self, current_price: float) -> float:

        """计算价格动量"""

        if len(self.price_buffer) < 5:
            return 0.0

        recent_prices = list(self.price_buffer)[-5:]

        if len(recent_prices) < 5:
            return 0.0

        price_changes = [(recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]

                         for i in range(1, len(recent_prices))]

        return np.mean(price_changes) if price_changes else 0.0

    def _analyze_volume_profile(self) -> Dict[str, float]:

        """分析成交量分布"""

        if not self.volume_buffer:
            return {'avg_volume': 0, 'volume_trend': 0}

        volumes = list(self.volume_buffer)

        avg_volume = np.mean(volumes) if volumes else 0

        # 计算成交量趋势

        if len(volumes) >= 10:

            recent_volumes = volumes[-10:]

            volume_trend = (np.mean(recent_volumes[-5:]) - np.mean(recent_volumes[:5])) / np.mean(

                recent_volumes[:5]) if np.mean(recent_volumes[:5]) > 0 else 0

        else:

            volume_trend = 0

        return {'avg_volume': avg_volume, 'volume_trend': volume_trend}

    def _update_data_buffers(self, tick_data: Dict[str, Any]):

        """更新数据缓冲区"""

        self.tick_buffer.append(tick_data)

        self.price_buffer.append(tick_data['mid_price'])

        if tick_data['volume'] > 0:
            self.volume_buffer.append(tick_data['volume'])

        # 更新高低点缓冲区

        self._update_high_low_buffers(tick_data['mid_price'])

    def _update_high_low_buffers(self, current_price: float):

        """更新高低点缓冲区"""

        if not self.high_buffer or current_price > self.high_buffer[-1]:

            self.high_buffer.append(current_price)

        else:

            self.high_buffer.append(self.high_buffer[-1])

        if not self.low_buffer or current_price < self.low_buffer[-1]:

            self.low_buffer.append(current_price)

        else:

            self.low_buffer.append(self.low_buffer[-1])

    def _update_data_quality_metrics(self, tick_data: Dict[str, Any]):

        """更新数据质量指标"""

        self.data_quality['total_ticks'] += 1

        self.data_quality['valid_ticks'] += 1

        # 更新平均点差

        spreads = [t['spread'] for t in list(self.tick_buffer)[-100:]]

        if spreads:
            self.data_quality['avg_spread'] = np.mean(spreads)

        # 更新Tick频率

        current_time = time.time()

        if self.data_quality.get('last_tick_time', 0) > 0:

            time_diff = current_time - self.data_quality['last_tick_time']

            if time_diff > 0:
                new_freq = 1.0 / time_diff

                self.data_quality['tick_frequency'] = (

                        0.9 * self.data_quality['tick_frequency'] + 0.1 * new_freq

                )

        self.data_quality['last_tick_time'] = current_time

    def _report_initialization_status(self):

        """报告初始化状态"""

        logger.info("=== 数据引擎初始化状态 ===")

        logger.info(f"品种: {self.symbol}")

        logger.info(f"缓冲区大小: {len(self.tick_buffer)}")

        logger.info(f"数据质量: {self.data_quality['valid_ticks']}/{self.data_quality['total_ticks']} "

                    f"({self.data_quality['valid_ticks'] / self.data_quality['total_ticks']:.1%})")

        logger.info(f"平均点差: {self.data_quality['avg_spread']:.1f}点")

        logger.info(f"Tick频率: {self.data_quality['tick_frequency']:.1f}Hz")

        logger.info("=========================")

    def calculate_complex_indicators(self) -> Dict[str, float]:

        """计算复杂技术指标 - 保持所有复杂性"""

        if not self.initialized:
            return {}

        try:

            indicators = {}

            prices = np.array(list(self.price_buffer))

            if len(prices) < 50:  # 最小数据要求

                return indicators

            current_price = prices[-1]

            indicators['CURRENT_PRICE'] = current_price

            # 1. 多周期RSI

            for period in ProfessionalComplexConfig.TECHNICAL_INDICATORS['RSI']['PERIODS']:

                if len(prices) >= period + 1:

                    try:

                        rsi = talib.RSI(prices, timeperiod=period)

                        if len(rsi) > 0 and not np.isnan(rsi[-1]):

                            rsi_value = float(rsi[-1])

                            if 0 <= rsi_value <= 100:

                                indicators[f'RSI_{period}'] = rsi_value

                            else:

                                indicators[f'RSI_{period}'] = 50

                        else:

                            indicators[f'RSI_{period}'] = 50

                    except Exception as e:

                        logger.warning(f"RSI{period}计算异常: {str(e)}")

                        indicators[f'RSI_{period}'] = 50

            # 2. MACD系列指标

            if len(prices) >= ProfessionalComplexConfig.TECHNICAL_INDICATORS['MACD']['SLOW']:

                try:

                    macd, macd_signal, macd_hist = talib.MACD(

                        prices,

                        fastperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['MACD']['FAST'],

                        slowperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['MACD']['SLOW'],

                        signalperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['MACD']['SIGNAL']

                    )

                    indicators['MACD'] = macd[-1] if not np.isnan(macd[-1]) else 0

                    indicators['MACD_SIGNAL'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0

                    indicators['MACD_HIST'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0

                    indicators['MACD_TREND'] = self._analyze_macd_trend(macd, macd_signal, macd_hist)

                except Exception as e:

                    logger.warning(f"MACD计算异常: {str(e)}")

            # 3. 多周期EMA分析

            ema_series = {}

            for period in ProfessionalComplexConfig.TECHNICAL_INDICATORS['EMA']['PERIODS']:

                if len(prices) >= period:

                    try:

                        ema = talib.EMA(prices, timeperiod=period)

                        if not np.isnan(ema[-1]):
                            indicators[f'EMA_{period}'] = ema[-1]

                            ema_series[period] = ema[-1]

                    except Exception as e:

                        logger.warning(f"EMA{period}计算异常: {str(e)}")

            # EMA排列分析

            if len(ema_series) >= 3:
                indicators['EMA_ALIGNMENT'] = self._analyze_ema_alignment(ema_series, current_price)

            # 4. 多标准差布林带

            bb_period = ProfessionalComplexConfig.TECHNICAL_INDICATORS['BOLLINGER']['PERIOD']

            if len(prices) >= bb_period:

                for std_dev in ProfessionalComplexConfig.TECHNICAL_INDICATORS['BOLLINGER']['BANDS']:

                    try:

                        upper, middle, lower = talib.BBANDS(

                            prices, timeperiod=bb_period,

                            nbdevup=std_dev, nbdevdn=std_dev

                        )

                        indicators[f'BB_UPPER_{std_dev}'] = upper[-1] if not np.isnan(upper[-1]) else current_price

                        indicators[f'BB_LOWER_{std_dev}'] = lower[-1] if not np.isnan(lower[-1]) else current_price

                    except Exception as e:

                        logger.warning(f"布林带计算异常(std_dev={std_dev}): {str(e)}")

                # 布林带综合分析

                indicators['BB_POSITION'] = self._analyze_bollinger_position(indicators, current_price)

                indicators['BB_WIDTH_RATIO'] = self._calculate_bollinger_width(indicators, current_price)

            # 5. 波动率指标

            if len(prices) >= ProfessionalComplexConfig.TECHNICAL_INDICATORS['ATR']['PERIOD']:

                try:

                    highs = np.array(list(self.high_buffer))

                    lows = np.array(list(self.low_buffer))

                    atr = talib.ATR(highs, lows, prices,

                                    timeperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['ATR']['PERIOD'])

                    indicators['ATR'] = atr[-1] if not np.isnan(atr[-1]) else current_price * 0.001

                    indicators['ATR_PERCENT'] = indicators['ATR'] / current_price if current_price > 0 else 0

                except Exception as e:

                    logger.warning(f"ATR计算异常: {str(e)}")

                    indicators['ATR'] = current_price * 0.001

            # 6. 趋势强度指标

            if len(prices) >= ProfessionalComplexConfig.TECHNICAL_INDICATORS['ADX']['PERIOD']:

                try:

                    highs = np.array(list(self.high_buffer))

                    lows = np.array(list(self.low_buffer))

                    adx = talib.ADX(highs, lows, prices,

                                    timeperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['ADX']['PERIOD'])

                    indicators['ADX'] = adx[-1] if not np.isnan(adx[-1]) else 0

                    # 附加趋势指标

                    plus_di = talib.PLUS_DI(highs, lows, prices,

                                            timeperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['ADX']['PERIOD'])

                    minus_di = talib.MINUS_DI(highs, lows, prices,

                                              timeperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['ADX'][

                                                  'PERIOD'])

                    indicators['PLUS_DI'] = plus_di[-1] if not np.isnan(plus_di[-1]) else 0

                    indicators['MINUS_DI'] = minus_di[-1] if not np.isnan(minus_di[-1]) else 0

                except Exception as e:

                    logger.warning(f"ADX计算异常: {str(e)}")

            # 7. 震荡指标

            if len(prices) >= ProfessionalComplexConfig.TECHNICAL_INDICATORS['STOCHASTIC']['K_PERIOD']:

                try:

                    highs = np.array(list(self.high_buffer))

                    lows = np.array(list(self.low_buffer))

                    stoch_k, stoch_d = talib.STOCH(highs, lows, prices,

                                                   fastk_period=

                                                   ProfessionalComplexConfig.TECHNICAL_INDICATORS['STOCHASTIC'][

                                                       'K_PERIOD'],

                                                   slowk_period=

                                                   ProfessionalComplexConfig.TECHNICAL_INDICATORS['STOCHASTIC'][

                                                       'SLOWING'],

                                                   slowd_period=

                                                   ProfessionalComplexConfig.TECHNICAL_INDICATORS['STOCHASTIC'][

                                                       'D_PERIOD'])

                    indicators['STOCH_K'] = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50

                    indicators['STOCH_D'] = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50

                    indicators['STOCH_CROSS'] = self._analyze_stochastic_cross(stoch_k, stoch_d)

                except Exception as e:

                    logger.warning(f"随机指标计算异常: {str(e)}")

            # 8. 其他高级指标

            if len(prices) >= 20:

                try:

                    # CCI商品通道指数

                    cci = talib.CCI(np.array(list(self.high_buffer)),

                                    np.array(list(self.low_buffer)), prices,

                                    timeperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['CCI']['PERIOD'])

                    indicators['CCI'] = cci[-1] if not np.isnan(cci[-1]) else 0

                    # 威廉指标

                    williams = talib.WILLR(np.array(list(self.high_buffer)),

                                           np.array(list(self.low_buffer)), prices,

                                           timeperiod=ProfessionalComplexConfig.TECHNICAL_INDICATORS['WILLIAMSR'][

                                               'PERIOD'])

                    indicators['WILLIAMSR'] = williams[-1] if not np.isnan(williams[-1]) else -50

                except Exception as e:

                    logger.warning(f"高级指标计算异常: {str(e)}")

            # 8. 多时间框架EMA分析
            try:
                timeframe_emas = self._calculate_timeframe_emas()
                if timeframe_emas:
                    indicators['TIMEFRAME_EMAS'] = timeframe_emas
                    # 检查EMA趋势排列
                    ema_trend_result = self._check_ema_trend_alignment(timeframe_emas)
                    indicators['EMA_TREND_ALIGNMENT'] = ema_trend_result
                    # 添加便捷访问
                    indicators['EMA_TREND'] = ema_trend_result.get('trend', 'UNCERTAIN')
                    indicators['EMA_TREND_TIMEFRAME'] = ema_trend_result.get('timeframe')
            except Exception as e:
                logger.warning(f"多时间框架EMA计算异常: {str(e)}")

            # 9. KDJ指标
            try:
                kdj = self._calculate_kdj_indicator(mt5.TIMEFRAME_M5)
                if kdj:
                    indicators['KDJ'] = kdj
                    indicators['KDJ_K'] = kdj.get('K', 50.0)
                    indicators['KDJ_D'] = kdj.get('D', 50.0)
                    indicators['KDJ_J'] = kdj.get('J', 50.0)
                    indicators['KDJ_GOLDEN_CROSS'] = kdj.get('GOLDEN_CROSS', False)
                    indicators['KDJ_DEATH_CROSS'] = kdj.get('DEATH_CROSS', False)
                    indicators['KDJ_OVERSOLD'] = kdj.get('OVERSOLD', False)
                    indicators['KDJ_OVERBOUGHT'] = kdj.get('OVERBOUGHT', False)
            except Exception as e:
                logger.warning(f"KDJ指标计算异常: {str(e)}")

            # 缓存计算结果

            self.indicators_cache = indicators.copy()

            # 保存历史指标（用于趋势启动检测）
            if 'ADX' in indicators:
                indicators['ADX_PREV'] = self.previous_indicators.get('ADX', 0)
            if 'MACD_HIST' in indicators:
                indicators['MACD_HIST_PREV'] = self.previous_indicators.get('MACD_HIST', 0)

            # 更新历史指标
            self.previous_indicators = indicators.copy()

            logger.debug(f"📊 计算完成: {len(indicators)}个技术指标")

            return indicators

        except Exception as e:

            logger.error(f"计算技术指标异常: {str(e)}")

            return {}

    def _analyze_macd_trend(self, macd: np.ndarray, macd_signal: np.ndarray, macd_hist: np.ndarray) -> float:

        """分析MACD趋势强度"""

        if len(macd) < 3:
            return 0.0

        # MACD在信号线上方且上升

        if macd[-1] > macd_signal[-1] and macd[-1] > macd[-2]:

            return 0.8  # 强看涨

        elif macd[-1] < macd_signal[-1] and macd[-1] < macd[-2]:

            return -0.8  # 强看跌

        elif macd[-1] > macd_signal[-1]:

            return 0.4  # 弱看涨

        elif macd[-1] < macd_signal[-1]:

            return -0.4  # 弱看跌

        else:

            return 0.0

    def _analyze_ema_alignment(self, ema_series: Dict[int, float], current_price: float) -> float:

        """分析EMA排列

        多头排列（上升趋势）：EMA5 > EMA15 > EMA30 > EMA60，且价格 > EMA60
        空头排列（下降趋势）：EMA5 < EMA15 < EMA30 < EMA60，且价格 < EMA60
        """

        periods = sorted(ema_series.keys())

        if len(periods) < 3:
            return 0.0

        # 检查多头排列：短期EMA > 长期EMA（EMA5 > EMA15 > EMA30 > EMA60）
        is_bullish = all(ema_series[periods[i]] > ema_series[periods[i + 1]] for i in range(len(periods) - 1))

        # 检查空头排列：短期EMA < 长期EMA（EMA5 < EMA15 < EMA30 < EMA60）
        is_bearish = all(ema_series[periods[i]] < ema_series[periods[i + 1]] for i in range(len(periods) - 1))

        if is_bullish and current_price > ema_series[periods[-1]]:

            return 0.9  # 强多头：EMA多头排列且价格在均线之上

        elif is_bearish and current_price < ema_series[periods[-1]]:

            return -0.9  # 强空头：EMA空头排列且价格在均线之下

        elif is_bullish:

            return 0.6  # 弱多头：EMA多头排列但价格未突破

        elif is_bearish:

            return -0.6  # 弱空头：EMA空头排列但价格未跌破

        else:

            return 0.0  # 混乱排列

    def _analyze_bollinger_position(self, indicators: Dict[str, float], current_price: float) -> float:

        """分析布林带位置"""

        bb_upper = indicators.get('BB_UPPER_2.0', current_price)

        bb_lower = indicators.get('BB_LOWER_2.0', current_price)

        if bb_upper == bb_lower:
            return 0.5

        position = (current_price - bb_lower) / (bb_upper - bb_lower)

        return max(0.0, min(1.0, position))

    def _calculate_bollinger_width(self, indicators: Dict[str, float], current_price: float) -> float:

        """计算布林带宽度比率"""

        bb_upper = indicators.get('BB_UPPER_2.0', current_price)

        bb_lower = indicators.get('BB_LOWER_2.0', current_price)

        if current_price == 0:
            return 0.0

        width = (bb_upper - bb_lower) / current_price

        return width

    def _analyze_stochastic_cross(self, stoch_k: np.ndarray, stoch_d: np.ndarray) -> float:

        """分析随机指标交叉"""

        if len(stoch_k) < 2 or len(stoch_d) < 2:
            return 0.0

        # 金叉

        if stoch_k[-1] > stoch_d[-1] and stoch_k[-2] <= stoch_d[-2]:

            return 0.7

        # 死叉

        elif stoch_k[-1] < stoch_d[-1] and stoch_k[-2] >= stoch_d[-2]:

            return -0.7

        else:

            return 0.0

    def _get_candle_data(self, timeframe: int, count: int = 100) -> Optional[np.ndarray]:
        """从MT5获取K线数据"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return None
            return rates
        except Exception as e:
            logger.warning(f"获取K线数据异常(timeframe={timeframe}): {str(e)}")
            return None

    def _calculate_timeframe_emas(self) -> Dict[str, Dict[str, float]]:
        """计算多时间框架EMA（5分钟、15分钟、30分钟、60分钟）"""
        timeframe_emas = {}
        timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'M60': mt5.TIMEFRAME_H1  # 使用H1代替M60（MT5没有TIMEFRAME_M60）
        }

        for tf_name, tf_value in timeframes.items():
            try:
                # 获取K线数据，需要足够的数据来计算EMA60
                rates = self._get_candle_data(tf_value, 100)
                if rates is None or len(rates) < 60:
                    continue

                # 提取收盘价
                closes = rates['close']

                # 计算各周期EMA
                ema_5 = talib.EMA(closes, timeperiod=5)
                ema_15 = talib.EMA(closes, timeperiod=15)
                ema_30 = talib.EMA(closes, timeperiod=30)
                ema_60 = talib.EMA(closes, timeperiod=60)

                # 获取最新值
                if len(ema_5) > 0 and not np.isnan(ema_5[-1]):
                    timeframe_emas[tf_name] = {
                        'MA5': float(ema_5[-1]),
                        'MA15': float(ema_15[-1]) if len(ema_15) > 0 and not np.isnan(ema_15[-1]) else None,
                        'MA30': float(ema_30[-1]) if len(ema_30) > 0 and not np.isnan(ema_30[-1]) else None,
                        'MA60': float(ema_60[-1]) if len(ema_60) > 0 and not np.isnan(ema_60[-1]) else None,
                        'CLOSE': float(closes[-1])
                    }
            except Exception as e:
                logger.warning(f"计算{tf_name}时间框架EMA异常: {str(e)}")
                continue

        return timeframe_emas

    def _check_ema_trend_alignment(self, timeframe_emas: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """检查多时间框架EMA趋势排列（用户标准）
        多头趋势：MA5 > MA15 > MA30 > MA60
        空头趋势：MA5 < MA15 < MA30 < MA60
        震荡市：其他情况
        """
        result = {
            'trend': 'UNCERTAIN',  # BULLISH, BEARISH, UNCERTAIN（震荡市）
            'timeframe': None,  # 哪个时间框架有明确趋势
            'details': {}
        }

        # 优先检查M5（5分钟）时间框架
        for tf_name in ['M5', 'M15', 'M30', 'M60']:
            if tf_name not in timeframe_emas:
                continue

            emas = timeframe_emas[tf_name]
            ma5 = emas.get('MA5')
            ma15 = emas.get('MA15')
            ma30 = emas.get('MA30')
            ma60 = emas.get('MA60')

            if None in [ma5, ma15, ma30, ma60]:
                continue

            # 检查多头排列：MA5 > MA15 > MA30 > MA60（严格按顺序）
            is_bullish = (ma5 > ma15 and ma15 > ma30 and ma30 > ma60)
            # 检查空头排列：MA5 < MA15 < MA30 < MA60（严格按顺序）
            is_bearish = (ma5 < ma15 and ma15 < ma30 and ma30 < ma60)

            # 添加详细日志用于调试
            current_time = time.time()
            if int(current_time) % 30 == 0:  # 每30秒输出一次EMA趋势判断详情
                logger.info(
                    f"📊 [{tf_name}] EMA趋势判断: MA5={ma5:.2f}, MA15={ma15:.2f}, MA30={ma30:.2f}, MA60={ma60:.2f}")
                logger.info(
                    f"   多头检查: MA5>MA15={ma5 > ma15}, MA15>MA30={ma15 > ma30}, MA30>MA60={ma30 > ma60}, 结果={is_bullish}")
                logger.info(
                    f"   空头检查: MA5<MA15={ma5 < ma15}, MA15<MA30={ma15 < ma30}, MA30<MA60={ma30 < ma60}, 结果={is_bearish}")

            if is_bullish:
                result['trend'] = 'BULLISH'
                result['timeframe'] = tf_name
                result['details'][tf_name] = {
                    'type': 'BULLISH',
                    'ma5': ma5,
                    'ma15': ma15,
                    'ma30': ma30,
                    'ma60': ma60
                }
                if int(current_time) % 30 == 0:
                    logger.info(
                        f"✅ [{tf_name}] 判断为多头趋势: MA5({ma5:.2f}) > MA15({ma15:.2f}) > MA30({ma30:.2f}) > MA60({ma60:.2f})")
                # 找到明确趋势后，优先返回M5的结果
                if tf_name == 'M5':
                    break

            elif is_bearish:
                result['trend'] = 'BEARISH'
                result['timeframe'] = tf_name
                result['details'][tf_name] = {
                    'type': 'BEARISH',
                    'ma5': ma5,
                    'ma15': ma15,
                    'ma30': ma30,
                    'ma60': ma60
                }
                if int(current_time) % 30 == 0:
                    logger.info(
                        f"✅ [{tf_name}] 判断为空头趋势: MA5({ma5:.2f}) < MA15({ma15:.2f}) < MA30({ma30:.2f}) < MA60({ma60:.2f})")
                # 找到明确趋势后，优先返回M5的结果
                if tf_name == 'M5':
                    break

        return result

    def _calculate_kdj_indicator(self, timeframe: int = mt5.TIMEFRAME_M5) -> Optional[Dict[str, float]]:
        """计算KDJ指标
        KDJ是基于随机指标（Stochastic）的改进版本
        RSV = (收盘价 - 最低价) / (最高价 - 最低价) * 100
        K = (2/3) * 前K值 + (1/3) * RSV
        D = (2/3) * 前D值 + (1/3) * K
        J = 3 * K - 2 * D
        """
        try:
            kdj_config = ProfessionalComplexConfig.TECHNICAL_INDICATORS['KDJ']
            rsv_period = kdj_config['RSV_PERIOD']
            k_period = kdj_config['K_PERIOD']
            d_period = kdj_config['D_PERIOD']

            # 获取K线数据
            rates = self._get_candle_data(timeframe, 100)
            if rates is None or len(rates) < rsv_period + 10:
                return None

            highs = rates['high']
            lows = rates['low']
            closes = rates['close']

            # 计算RSV
            rsv_values = []
            for i in range(rsv_period - 1, len(closes)):
                period_high = np.max(highs[i - rsv_period + 1:i + 1])
                period_low = np.min(lows[i - rsv_period + 1:i + 1])
                if period_high != period_low:
                    rsv = ((closes[i] - period_low) / (period_high - period_low)) * 100
                else:
                    rsv = 50.0  # 避免除零
                rsv_values.append(rsv)

            if len(rsv_values) < k_period + d_period:
                return None

            # 计算K值（使用EMA平滑）
            k_values = []
            # 初始K值
            k_prev = 50.0  # 初始值设为50
            for rsv in rsv_values:
                k = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * rsv
                k_values.append(k)
                k_prev = k

            # 计算D值（对K值进行EMA平滑）
            d_values = []
            d_prev = 50.0  # 初始值设为50
            for k in k_values:
                d = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k
                d_values.append(d)
                d_prev = d

            # 计算J值
            j_values = []
            for i in range(len(k_values)):
                if i < len(d_values):
                    j = 3 * k_values[i] - 2 * d_values[i]
                    j_values.append(j)

            # 获取最新值
            if len(k_values) > 0 and len(d_values) > 0 and len(j_values) > 0:
                k_current = k_values[-1]
                d_current = d_values[-1]
                j_current = j_values[-1] if len(j_values) > 0 else (3 * k_current - 2 * d_current)

                # 判断金叉死叉
                golden_cross = False
                death_cross = False
                if len(k_values) >= 2 and len(d_values) >= 2:
                    # 金叉：K向上穿越D
                    if k_values[-1] > d_values[-1] and k_values[-2] <= d_values[-2]:
                        golden_cross = True
                    # 死叉：K向下穿越D
                    elif k_values[-1] < d_values[-1] and k_values[-2] >= d_values[-2]:
                        death_cross = True

                return {
                    'K': float(k_current),
                    'D': float(d_current),
                    'J': float(j_current),
                    'GOLDEN_CROSS': golden_cross,
                    'DEATH_CROSS': death_cross,
                    'OVERSOLD': k_current < 20 and d_current < 20,  # 超卖
                    'OVERBOUGHT': k_current > 80 and d_current > 80  # 超买
                }

            return None

        except Exception as e:
            logger.warning(f"计算KDJ指标异常: {str(e)}")
            return None

    def get_multi_timeframe_analysis(self) -> Dict[str, Dict[str, float]]:

        """获取多时间框架分析"""

        if not self.initialized:
            return {}

        analysis = {}

        prices = np.array(list(self.price_buffer))

        for tf_name, tf_ticks in ProfessionalComplexConfig.TICK_TIMEFRAMES.items():

            if len(prices) >= tf_ticks:

                tf_prices = prices[-tf_ticks:]

                # 计算时间框架特定指标

                tf_indicators = {}

                # 简化的时间框架分析

                if len(tf_prices) > 0:
                    price_change = (tf_prices[-1] - tf_prices[0]) / tf_prices[0] if tf_prices[0] > 0 else 0

                    tf_indicators['PRICE_CHANGE'] = price_change

                    # 波动率

                    volatility = np.std(tf_prices) / np.mean(tf_prices) if np.mean(tf_prices) > 0 else 0

                    tf_indicators['VOLATILITY'] = volatility

                analysis[tf_name] = tf_indicators

        return analysis


class AdvancedMarketStateAnalyzer:
    """高级市场状态分析器 - 保持复杂性"""

    def __init__(self, data_engine: ProfessionalTickDataEngine):

        self.data_engine = data_engine

        self.current_state = "UNCERTAIN"

        self.state_confidence = 0.0

        self.state_duration = 0

        self.last_state_change = time.time()

        self.state_history = deque(maxlen=50)

    def analyze_complex_market_state(self) -> Tuple[str, float]:

        """分析复杂市场状态"""

        if not self.data_engine.initialized:
            return "UNCERTAIN", 0.0

        try:

            indicators = self.data_engine.calculate_complex_indicators()

            if not indicators:
                return "UNCERTAIN", 0.0

            # 多维度状态概率计算

            raw_probabilities = {

                'TRENDING': self._calculate_trending_probability(indicators),

                'RANGING': self._calculate_ranging_probability(indicators),

                'VOLATILE': self._calculate_volatile_probability(indicators),

                'UNCERTAIN': 0.1  # 基础不确定性

            }

            # 添加诊断日志（降低频率）
            current_time = time.time()
            if int(current_time) % 60 == 0:  # 每60秒记录一次原始概率
                logger.info(f"🔍 原始概率: TRENDING={raw_probabilities['TRENDING']:.3f}, "
                            f"RANGING={raw_probabilities['RANGING']:.3f}, "
                            f"VOLATILE={raw_probabilities['VOLATILE']:.3f}, "
                            f"UNCERTAIN={raw_probabilities['UNCERTAIN']:.3f}")

            # 使用改进的归一化方法 - 优化以提高置信度
            # 如果所有原始概率都很低，直接使用原始概率（不进行softmax）
            max_raw_prob = max(raw_probabilities.values())

            if max_raw_prob > 0.1:
                # 如果最高概率 > 0.1，使用softmax归一化
                # 提高温度参数，使分布更平滑，避免过度集中
                temperature = 1.5  # 从1.2提高到1.5，使分布更平滑
                exp_probs = {}
                for state, prob in raw_probabilities.items():
                    # 限制概率范围在0-1之间
                    prob = max(0.0, min(1.0, prob))
                    # 使用偏移，避免所有概率都很低时softmax失效
                    # 增加偏移量，提高高概率状态的权重
                    offset = 0.15 if prob > 0.5 else 0.1
                    exp_probs[state] = math.exp((prob + offset) / temperature)

                sum_exp = sum(exp_probs.values())
                if sum_exp > 0:
                    state_probabilities = {k: v / sum_exp for k, v in exp_probs.items()}
                else:
                    # 如果所有概率都为0，使用均匀分布
                    state_probabilities = {k: 0.25 for k in raw_probabilities.keys()}
            else:
                # 如果所有原始概率都很低，使用加权归一化（给高概率更多权重）
                total_raw = sum(raw_probabilities.values())
                if total_raw > 0:
                    # 使用平方根加权，提高高概率状态的权重
                    weighted_probs = {k: v ** 0.7 for k, v in raw_probabilities.items()}
                    total_weighted = sum(weighted_probs.values())
                    if total_weighted > 0:
                        state_probabilities = {k: v / total_weighted for k, v in weighted_probs.items()}
                    else:
                        state_probabilities = {k: v / total_raw for k, v in raw_probabilities.items()}
                else:
                    # 如果所有概率都为0，使用均匀分布
                    state_probabilities = {k: 0.25 for k in raw_probabilities.keys()}

            # 选择最可能的状态

            max_state = max(state_probabilities, key=state_probabilities.get)

            max_prob = state_probabilities[max_state]

            # 添加诊断日志（降低频率）
            if int(current_time) % 60 == 0:  # 每60秒记录一次归一化后的概率
                logger.info(f"🔍 归一化后概率: TRENDING={state_probabilities['TRENDING']:.3f}, "
                            f"RANGING={state_probabilities['RANGING']:.3f}, "
                            f"VOLATILE={state_probabilities['VOLATILE']:.3f}, "
                            f"UNCERTAIN={state_probabilities['UNCERTAIN']:.3f}, "
                            f"最高状态: {max_state} (概率: {max_prob:.3f})")

            # 状态转换逻辑 - 增强稳定性
            min_state_duration = 10.0  # 增加到10秒，避免频繁切换
            state_duration = time.time() - self.last_state_change

            # 调整状态转换阈值（根据归一化后的概率范围调整）
            # 如果使用softmax，概率会更分散；如果直接归一化，概率会更集中
            state_change_threshold = 0.4  # 降低阈值，因为归一化后概率可能较低
            current_state_prob = state_probabilities.get(self.current_state, 0)
            prob_difference = max_prob - current_state_prob

            # 计算第二高概率，确保新状态明显优于其他所有状态
            sorted_probs = sorted(state_probabilities.values(), reverse=True)
            second_max_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
            prob_advantage = max_prob - second_max_prob  # 与第二高概率的差值

            # 增强的转换条件：
            # 1. 新状态概率 > 阈值（降低到0.4）
            # 2. 新状态概率明显高于当前状态（差值 > 0.15，降低要求）
            # 3. 当前状态持续时间 >= 最小持续时间（10秒）
            # 4. 新状态概率必须明显高于其他所有状态（差值 > 0.10，降低要求）
            # 特殊处理：如果当前状态是UNCERTAIN，降低转换要求
            if self.current_state == 'UNCERTAIN':
                # 从UNCERTAIN转换时，降低要求
                min_state_duration_uncertain = 5.0  # UNCERTAIN状态只需持续5秒
                prob_difference_threshold = 0.10  # 降低差值要求
                prob_advantage_threshold = 0.05  # 降低优势要求
                should_change = (
                        max_prob > 0.3 and  # 降低阈值
                        max_state != self.current_state and
                        prob_difference > prob_difference_threshold and
                        state_duration >= min_state_duration_uncertain and
                        prob_advantage > prob_advantage_threshold
                )
            else:
                should_change = (
                        max_prob > state_change_threshold and
                        max_state != self.current_state and
                        prob_difference > 0.15 and  # 降低到0.15
                        state_duration >= min_state_duration and  # 至少持续10秒
                        prob_advantage > 0.10  # 降低到0.10
                )

            if should_change:

                old_state = self.current_state

                self.current_state = max_state

                # 优化置信度计算：考虑概率优势
                # 如果最高概率明显高于其他状态，提高置信度
                if prob_advantage > 0.15:
                    self.state_confidence = min(1.0, max_prob * 1.15)  # 优势明显，提高15%
                elif prob_advantage > 0.10:
                    self.state_confidence = min(1.0, max_prob * 1.10)  # 优势中等，提高10%
                else:
                    self.state_confidence = max_prob

                self.last_state_change = time.time()

                self.state_duration = 0

                # 记录状态变更

                state_record = {

                    'timestamp': time.time(),

                    'from_state': old_state,

                    'to_state': max_state,

                    'confidence': max_prob,

                    'duration': state_duration

                }

                self.state_history.append(state_record)

                logger.info(
                    f"🔄 市场状态变更: {old_state} -> {max_state} (置信度: {max_prob:.2f}, 持续时间: {state_duration:.1f}秒)")
                logger.debug(f"   概率分布: TRENDING={state_probabilities['TRENDING']:.2f}, "
                             f"RANGING={state_probabilities['RANGING']:.2f}, "
                             f"VOLATILE={state_probabilities['VOLATILE']:.2f}, "
                             f"UNCERTAIN={state_probabilities['UNCERTAIN']:.2f}")

            else:

                self.state_duration = state_duration
                # 无论状态是否变更，都要更新置信度为当前最高概率
                # 优化置信度计算：考虑概率优势
                if prob_advantage > 0.15:
                    self.state_confidence = min(1.0, max_prob * 1.15)  # 优势明显，提高15%
                elif prob_advantage > 0.10:
                    self.state_confidence = min(1.0, max_prob * 1.10)  # 优势中等，提高10%
                else:
                    self.state_confidence = max_prob

                # 如果状态未变更，但概率接近，记录调试信息
                if max_state != self.current_state:
                    if prob_difference > 0.1:
                        logger.debug(f"⏸️ 状态未变更: {self.current_state} (当前概率: {current_state_prob:.2f}, "
                                     f"最高概率: {max_prob:.2f}, 差值: {prob_difference:.2f}, "
                                     f"持续时间: {state_duration:.1f}秒, 优势: {prob_advantage:.2f})")
                    # 如果当前状态是UNCERTAIN，且最高概率明显高于UNCERTAIN，降低转换要求
                    elif self.current_state == 'UNCERTAIN' and max_prob > 0.3:
                        logger.info(f"🔄 检测到从UNCERTAIN转换到{max_state}的机会 (概率: {max_prob:.2f}, "
                                    f"差值: {prob_difference:.2f}, 持续时间: {state_duration:.1f}秒)")

            return self.current_state, self.state_confidence

        except Exception as e:

            logger.error(f"分析市场状态异常: {str(e)}")

            return "UNCERTAIN", 0.0

    def _calculate_trending_probability(self, indicators: Dict) -> float:

        """计算趋势市概率 - 优先使用多时间框架EMA排列"""

        probability = 0.0

        weight_sum = 0.0

        try:

            ema_trend = indicators.get('EMA_TREND', 'UNCERTAIN')

            if ema_trend in ['BULLISH', 'BEARISH']:
                # 有明确的EMA排列趋势，给予高概率
                ema_probability = 0.75  # 固定概率0.75（明确趋势）
                probability += ema_probability * 0.40  # 给予更高权重
                weight_sum += 0.40

                # 如果EMA趋势明确，其他指标作为确认
                # ADX趋势强度（确认）
                adx = indicators.get('ADX', 0)
                if adx > ProfessionalComplexConfig.MARKET_STATE_PARAMS['TRENDING']['ADX_THRESHOLD']:
                    adx_score = min(1.0, adx / 50.0)
                    probability += adx_score * 0.20
                    weight_sum += 0.20

                # MACD趋势确认
                macd_trend = indicators.get('MACD_TREND', 0)
                if abs(macd_trend) > 0.3:
                    probability += abs(macd_trend) * 0.20
                    weight_sum += 0.20

                # DI指标确认
                plus_di = indicators.get('PLUS_DI', 0)
                minus_di = indicators.get('MINUS_DI', 0)
                if (ema_trend == 'BULLISH' and plus_di > minus_di and plus_di > 25) or \
                        (ema_trend == 'BEARISH' and minus_di > plus_di and minus_di > 25):
                    probability += 0.20
                    weight_sum += 0.20

            else:
                # EMA排列不明确，视为震荡市
                # ADX趋势强度
                adx = indicators.get('ADX', 0)
                if adx > ProfessionalComplexConfig.MARKET_STATE_PARAMS['TRENDING']['ADX_THRESHOLD']:
                    adx_score = min(1.0, adx / 50.0)
                    probability += adx_score * 0.25
                    weight_sum += 0.25

                # EMA排列趋势（原逻辑）
                ema_alignment = indicators.get('EMA_ALIGNMENT', 0)
                if abs(ema_alignment) > 0.5:
                    alignment_score = abs(ema_alignment)
                    probability += alignment_score * 0.25
                    weight_sum += 0.25

                # MACD趋势确认
                macd_trend = indicators.get('MACD_TREND', 0)
                if abs(macd_trend) > 0.3:
                    probability += abs(macd_trend) * 0.20
                    weight_sum += 0.20

                # 价格动量
                prices = list(self.data_engine.price_buffer)
                if len(prices) >= 20:
                    momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
                    if abs(momentum_10) > ProfessionalComplexConfig.MARKET_STATE_PARAMS['TRENDING']['PRICE_MOMENTUM']:
                        momentum_score = min(1.0, abs(momentum_10) / 0.01)
                        probability += momentum_score * 0.15
                        weight_sum += 0.15

                # DI指标确认
                plus_di = indicators.get('PLUS_DI', 0)
                minus_di = indicators.get('MINUS_DI', 0)
                if plus_di > minus_di and plus_di > 25:
                    probability += 0.15
                    weight_sum += 0.15

            return probability / weight_sum if weight_sum > 0 else 0.0

        except Exception as e:

            logger.warning(f"计算趋势概率异常: {str(e)}")

            return 0.0

    def _calculate_ranging_probability(self, indicators: Dict) -> float:

        """计算震荡市概率 - 修复条件重叠问题"""

        probability = 0.0

        weight_sum = 0.0

        try:

            # 低波动率 - 修复与VOLATILE的重叠问题
            atr_percent = indicators.get('ATR_PERCENT', 0)
            atr_ranging_max = ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['ATR_RATIO_MAX']
            atr_volatile_min = ProfessionalComplexConfig.MARKET_STATE_PARAMS['VOLATILE']['ATR_RATIO_MIN']

            # 明确区分：RANGING要求ATR明显低于VOLATILE阈值
            if atr_percent < atr_ranging_max:
                # ATR在RANGING范围内
                low_vol_score = 1.0 - (atr_percent / atr_ranging_max)
                probability += low_vol_score * 0.30
                weight_sum += 0.30
            elif atr_percent > atr_volatile_min:
                # ATR在VOLATILE范围内，降低RANGING概率
                probability -= 0.20  # 负贡献
                weight_sum += 0.20
            else:
                # ATR在中间区域（0.0004-0.0006），给予较低的RANGING概率
                # 计算到RANGING阈值的距离
                distance_to_ranging = (atr_percent - atr_ranging_max) / (atr_volatile_min - atr_ranging_max) if (
                                                                                                                            atr_volatile_min - atr_ranging_max) > 0 else 0.5
                low_vol_score = max(0.0, 1.0 - distance_to_ranging * 2)  # 距离越远，分数越低
                probability += low_vol_score * 0.15  # 降低权重
                weight_sum += 0.15

            # 布林带收缩

            bb_width = indicators.get('BB_WIDTH_RATIO', 0)

            if bb_width < ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['BB_WIDTH_RATIO']:

                bb_score = 1.0 - (bb_width / ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['BB_WIDTH_RATIO'])

                probability += bb_score * 0.25

                weight_sum += 0.25
            elif bb_width > 0.003:
                # 布林带扩张，降低RANGING概率
                probability -= 0.10
                weight_sum += 0.10

            # ADX低值

            adx = indicators.get('ADX', 0)

            if adx < 20:

                adx_score = 1.0 - (adx / 20.0)

                probability += adx_score * 0.20

                weight_sum += 0.20
            else:
                # ADX高值，降低RANGING概率
                probability -= 0.10
                weight_sum += 0.10

            # 价格震荡模式

            prices = list(self.data_engine.price_buffer)

            if len(prices) >= 30:

                # 计算价格震荡幅度

                recent_high = max(prices[-15:])

                recent_low = min(prices[-15:])

                oscillation = (recent_high - recent_low) / ((recent_high + recent_low) / 2) if (

                                                                                                       recent_high + recent_low) > 0 else 0

                if oscillation < ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['PRICE_OSCILLATION']:

                    oscillation_score = 1.0 - (oscillation / ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING'][

                        'PRICE_OSCILLATION'])

                    probability += oscillation_score * 0.25

                    weight_sum += 0.25
                else:
                    # 震荡幅度大，降低RANGING概率
                    probability -= 0.10
                    weight_sum += 0.10

            # 限制概率范围在0-1之间，并归一化
            if weight_sum > 0:
                probability = probability / weight_sum
                probability = max(0.0, min(1.0, probability))  # 限制在0-1之间
            else:
                probability = 0.0

            return probability

        except Exception as e:

            logger.warning(f"计算震荡概率异常: {str(e)}")

            return 0.0

    def _calculate_volatile_probability(self, indicators: Dict) -> float:

        """计算高波动市概率 - 修复条件重叠问题"""

        probability = 0.0

        weight_sum = 0.0

        try:

            # 高波动率 - 修复与RANGING的重叠问题
            atr_percent = indicators.get('ATR_PERCENT', 0)
            atr_ranging_max = ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['ATR_RATIO_MAX']
            atr_volatile_min = ProfessionalComplexConfig.MARKET_STATE_PARAMS['VOLATILE']['ATR_RATIO_MIN']

            # 明确区分：VOLATILE要求ATR明显高于RANGING阈值
            if atr_percent > atr_volatile_min:
                # ATR在VOLATILE范围内
                high_vol_score = min(1.0, atr_percent / 0.001)
                probability += high_vol_score * 0.35
                weight_sum += 0.35
            elif atr_percent < atr_ranging_max:
                # ATR在RANGING范围内，降低VOLATILE概率
                probability -= 0.25  # 负贡献
                weight_sum += 0.25
            else:
                # ATR在中间区域（0.0004-0.0006），给予较低的VOLATILE概率
                # 计算到VOLATILE阈值的距离
                distance_to_volatile = (atr_volatile_min - atr_percent) / (atr_volatile_min - atr_ranging_max) if (
                                                                                                                              atr_volatile_min - atr_ranging_max) > 0 else 0.5
                high_vol_score = max(0.0, 1.0 - distance_to_volatile * 2)  # 距离越远，分数越低
                probability += high_vol_score * 0.15  # 降低权重
                weight_sum += 0.15

            # 布林带扩张

            bb_width = indicators.get('BB_WIDTH_RATIO', 0)

            if bb_width > 0.003:

                width_score = min(1.0, bb_width / 0.005)

                probability += width_score * 0.25

                weight_sum += 0.25
            elif bb_width < ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['BB_WIDTH_RATIO']:
                # 布林带收缩，降低VOLATILE概率
                probability -= 0.15
                weight_sum += 0.15

            # 价格大幅变动

            prices = list(self.data_engine.price_buffer)

            if len(prices) >= 10:

                max_change = max(

                    abs((prices[i] - prices[i - 1]) / prices[i - 1]) for i in range(1, min(10, len(prices))))

                if max_change > ProfessionalComplexConfig.MARKET_STATE_PARAMS['VOLATILE']['PRICE_SPIKE_FREQUENCY']:

                    change_score = min(1.0, max_change / 0.005)

                    probability += change_score * 0.25

                    weight_sum += 0.25
                else:
                    # 价格变动小，降低VOLATILE概率
                    probability -= 0.10
                    weight_sum += 0.10

            # 成交量异常

            volume_profile = self.data_engine.volume_buffer

            if len(volume_profile) >= 10:

                recent_volumes = list(volume_profile)[-10:]

                avg_volume = np.mean(recent_volumes) if recent_volumes else 0

                if avg_volume > 0:

                    volume_spike = max(recent_volumes) / avg_volume

                    if volume_spike > ProfessionalComplexConfig.MARKET_STATE_PARAMS['VOLATILE']['VOLUME_SPIKE_RATIO']:

                        probability += 0.15

                        weight_sum += 0.15
                    else:
                        # 成交量正常，降低VOLATILE概率
                        probability -= 0.05
                        weight_sum += 0.05

            # 限制概率范围在0-1之间，并归一化
            if weight_sum > 0:
                probability = probability / weight_sum
                probability = max(0.0, min(1.0, probability))  # 限制在0-1之间
            else:
                probability = 0.0

            return probability

        except Exception as e:

            logger.warning(f"计算波动概率异常: {str(e)}")

            return 0.0


class TechnicalPatternRecognizer:
    """技术形态识别器 - 识别各种K线形态和价格模式"""

    def __init__(self, data_engine: ProfessionalTickDataEngine):
        self.data_engine = data_engine
        self.pattern_cache = {}
        self.last_pattern_check = 0

    def detect_patterns(self, prices: List[float], highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """检测技术形态"""
        if len(prices) < 20:
            return {}

        patterns = {}

        # 1. 双顶/双底形态
        double_pattern = self._detect_double_top_bottom(prices, highs, lows)
        if double_pattern:
            patterns.update(double_pattern)

        # 2. 头肩顶/头肩底形态
        head_shoulder = self._detect_head_shoulders(prices, highs, lows)
        if head_shoulder:
            patterns.update(head_shoulder)

        # 3. 三角形形态（上升/下降/对称）
        triangle = self._detect_triangle(prices, highs, lows)
        if triangle:
            patterns.update(triangle)

        # 4. 旗形/矩形形态
        flag_pattern = self._detect_flag_rectangle(prices, highs, lows)
        if flag_pattern:
            patterns.update(flag_pattern)

        # 5. 支撑/阻力突破
        support_resistance = self._detect_support_resistance_breakout(prices, highs, lows)
        if support_resistance:
            patterns.update(support_resistance)

        # 6. 楔形形态
        wedge = self._detect_wedge(prices, highs, lows)
        if wedge:
            patterns.update(wedge)

        return patterns

    def _detect_double_top_bottom(self, prices: List[float], highs: List[float], lows: List[float]) -> Optional[Dict]:
        """检测双顶/双底形态"""
        if len(highs) < 20 or len(lows) < 20:
            return None

        # 寻找两个相近的高点（双顶）或低点（双底）
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]

        # 双顶检测
        if len(recent_highs) >= 10:
            # 找到最高点和次高点
            sorted_highs = sorted(enumerate(recent_highs), key=lambda x: x[1], reverse=True)
            if len(sorted_highs) >= 2:
                idx1, high1 = sorted_highs[0]
                idx2, high2 = sorted_highs[1]

                # 检查两个高点是否相近（差异<2%）
                if abs(high1 - high2) / max(high1, high2) < 0.02 and abs(idx1 - idx2) >= 5:
                    # 检查中间是否有明显的回撤
                    mid_range = recent_highs[min(idx1, idx2):max(idx1, idx2) + 1]
                    if mid_range:
                        mid_low = min(mid_range)
                        retracement = (max(high1, high2) - mid_low) / max(high1, high2)
                        if retracement > 0.03:  # 回撤至少3%
                            return {
                                'DOUBLE_TOP': {
                                    'type': 'BEARISH',
                                    'strength': min(1.0, retracement * 10),
                                    'resistance': max(high1, high2)
                                }
                            }

        # 双底检测
        if len(recent_lows) >= 10:
            sorted_lows = sorted(enumerate(recent_lows), key=lambda x: x[1])
            if len(sorted_lows) >= 2:
                idx1, low1 = sorted_lows[0]
                idx2, low2 = sorted_lows[1]

                if abs(low1 - low2) / max(low1, low2) < 0.02 and abs(idx1 - idx2) >= 5:
                    mid_range = recent_lows[min(idx1, idx2):max(idx1, idx2) + 1]
                    if mid_range:
                        mid_high = max(mid_range)
                        retracement = (mid_high - min(low1, low2)) / min(low1, low2)
                        if retracement > 0.03:
                            return {
                                'DOUBLE_BOTTOM': {
                                    'type': 'BULLISH',
                                    'strength': min(1.0, retracement * 10),
                                    'support': min(low1, low2)
                                }
                            }

        return None

    def _detect_head_shoulders(self, prices: List[float], highs: List[float], lows: List[float]) -> Optional[Dict]:
        """检测头肩顶/头肩底形态"""
        if len(highs) < 15 or len(lows) < 15:
            return None

        recent_highs = highs[-15:]
        recent_lows = lows[-15:]

        # 头肩顶：左肩-头-右肩，头最高
        if len(recent_highs) >= 10:
            # 简化检测：寻找三个高点，中间最高
            peaks = []
            for i in range(1, len(recent_highs) - 1):
                if recent_highs[i] > recent_highs[i - 1] and recent_highs[i] > recent_highs[i + 1]:
                    peaks.append((i, recent_highs[i]))

            if len(peaks) >= 3:
                # 检查中间峰值是否最高
                peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                if len(peaks_sorted) >= 3:
                    head_idx, head_high = peaks_sorted[0]
                    # 检查左右肩是否相近且低于头
                    shoulders = [p for p in peaks_sorted[1:] if abs(p[0] - head_idx) > 2]
                    if len(shoulders) >= 2:
                        left_shoulder = min(shoulders, key=lambda x: abs(x[0] - (head_idx - 5)))
                        right_shoulder = min(shoulders, key=lambda x: abs(x[0] - (head_idx + 5)))
                        if (head_high > left_shoulder[1] and head_high > right_shoulder[1] and
                                abs(left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1],
                                                                                right_shoulder[1]) < 0.03):
                            return {
                                'HEAD_SHOULDER_TOP': {
                                    'type': 'BEARISH',
                                    'strength': 0.7,
                                    'neckline': (left_shoulder[1] + right_shoulder[1]) / 2
                                }
                            }

        # 头肩底：左肩-头-右肩，头最低
        if len(recent_lows) >= 10:
            valleys = []
            for i in range(1, len(recent_lows) - 1):
                if recent_lows[i] < recent_lows[i - 1] and recent_lows[i] < recent_lows[i + 1]:
                    valleys.append((i, recent_lows[i]))

            if len(valleys) >= 3:
                valleys_sorted = sorted(valleys, key=lambda x: x[1])
                if len(valleys_sorted) >= 3:
                    head_idx, head_low = valleys_sorted[0]
                    shoulders = [v for v in valleys_sorted[1:] if abs(v[0] - head_idx) > 2]
                    if len(shoulders) >= 2:
                        left_shoulder = min(shoulders, key=lambda x: abs(x[0] - (head_idx - 5)))
                        right_shoulder = min(shoulders, key=lambda x: abs(x[0] - (head_idx + 5)))
                        if (head_low < left_shoulder[1] and head_low < right_shoulder[1] and
                                abs(left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1],
                                                                                right_shoulder[1]) < 0.03):
                            return {
                                'HEAD_SHOULDER_BOTTOM': {
                                    'type': 'BULLISH',
                                    'strength': 0.7,
                                    'neckline': (left_shoulder[1] + right_shoulder[1]) / 2
                                }
                            }

        return None

    def _detect_triangle(self, prices: List[float], highs: List[float], lows: List[float]) -> Optional[Dict]:
        """检测三角形形态（上升/下降/对称）"""
        if len(highs) < 10 or len(lows) < 10:
            return None

        recent_highs = highs[-10:]
        recent_lows = lows[-10:]

        # 计算高点和低点的趋势
        high_trend = (recent_highs[-1] - recent_highs[0]) / recent_highs[0] if recent_highs[0] > 0 else 0
        low_trend = (recent_lows[-1] - recent_lows[0]) / recent_lows[0] if recent_lows[0] > 0 else 0

        # 计算波动率收缩
        early_range = max(recent_highs[:5]) - min(recent_lows[:5])
        late_range = max(recent_highs[-5:]) - min(recent_lows[-5:])
        contraction = (early_range - late_range) / early_range if early_range > 0 else 0

        if contraction > 0.2:  # 波动率收缩至少20%
            # 上升三角形：高点水平，低点上升
            if abs(high_trend) < 0.01 and low_trend > 0.01:
                return {
                    'ASCENDING_TRIANGLE': {
                        'type': 'BULLISH',
                        'strength': min(1.0, contraction * 2),
                        'resistance': max(recent_highs)
                    }
                }
            # 下降三角形：低点水平，高点下降
            elif abs(low_trend) < 0.01 and high_trend < -0.01:
                return {
                    'DESCENDING_TRIANGLE': {
                        'type': 'BEARISH',
                        'strength': min(1.0, contraction * 2),
                        'support': min(recent_lows)
                    }
                }
            # 对称三角形：高点和低点都收敛
            elif abs(high_trend) < 0.015 and abs(low_trend) < 0.015:
                return {
                    'SYMMETRIC_TRIANGLE': {
                        'type': 'NEUTRAL',
                        'strength': min(1.0, contraction * 2),
                        'breakout_direction': 'UNKNOWN'
                    }
                }

        return None

    def _detect_flag_rectangle(self, prices: List[float], highs: List[float], lows: List[float]) -> Optional[Dict]:
        """检测旗形/矩形形态"""
        if len(prices) < 15:
            return None

        recent_prices = prices[-15:]
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]

        # 矩形：价格在水平区间内震荡
        price_range = max(recent_highs) - min(recent_lows)
        avg_price = sum(recent_prices) / len(recent_prices)
        range_ratio = price_range / avg_price if avg_price > 0 else 0

        if range_ratio < 0.02:  # 窄幅震荡
            # 检查是否有明显的趋势前导
            if len(prices) >= 20:
                prior_trend = (prices[-15] - prices[-20]) / prices[-20] if prices[-20] > 0 else 0
                if abs(prior_trend) > 0.01:  # 有明显的前导趋势
                    return {
                        'FLAG_PATTERN': {
                            'type': 'BULLISH' if prior_trend > 0 else 'BEARISH',
                            'strength': 0.6,
                            'continuation': True
                        }
                    }
                else:
                    return {
                        'RECTANGLE': {
                            'type': 'NEUTRAL',
                            'strength': 0.5,
                            'resistance': max(recent_highs),
                            'support': min(recent_lows)
                        }
                    }

        return None

    def _detect_support_resistance_breakout(self, prices: List[float], highs: List[float], lows: List[float]) -> \
    Optional[Dict]:
        """检测支撑/阻力突破"""
        if len(prices) < 20:
            return None

        recent_prices = prices[-20:]
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]

        current_price = recent_prices[-1]

        # 识别关键支撑和阻力位
        resistance = max(recent_highs[:-5])  # 排除最近5个点
        support = min(recent_lows[:-5])

        # 检查是否突破阻力
        if current_price > resistance * 0.998:
            breakout_strength = (current_price - resistance) / resistance if resistance > 0 else 0
            if breakout_strength > 0.0005:  # 突破至少0.05%
                return {
                    'RESISTANCE_BREAKOUT': {
                        'type': 'BULLISH',
                        'strength': min(1.0, breakout_strength * 100),
                        'resistance': resistance
                    }
                }

        # 检查是否跌破支撑
        if current_price < support * 1.002:
            breakdown_strength = (support - current_price) / support if support > 0 else 0
            if breakdown_strength > 0.0005:
                return {
                    'SUPPORT_BREAKDOWN': {
                        'type': 'BEARISH',
                        'strength': min(1.0, breakdown_strength * 100),
                        'support': support
                    }
                }

        return None

    def _detect_wedge(self, prices: List[float], highs: List[float], lows: List[float]) -> Optional[Dict]:
        """检测楔形形态"""
        if len(highs) < 10 or len(lows) < 10:
            return None

        recent_highs = highs[-10:]
        recent_lows = lows[-10:]

        # 计算高点和低点的趋势
        high_trend = (recent_highs[-1] - recent_highs[0]) / recent_highs[0] if recent_highs[0] > 0 else 0
        low_trend = (recent_lows[-1] - recent_lows[0]) / recent_lows[0] if recent_lows[0] > 0 else 0

        # 上升楔形：高点和低点都上升，但高点上升更快（看跌）
        if high_trend > 0.01 and low_trend > 0.01 and high_trend > low_trend * 1.2:
            return {
                'RISING_WEDGE': {
                    'type': 'BEARISH',
                    'strength': 0.6
                }
            }

        # 下降楔形：高点和低点都下降，但低点下降更快（看涨）
        if high_trend < -0.01 and low_trend < -0.01 and abs(low_trend) > abs(high_trend) * 1.2:
            return {
                'FALLING_WEDGE': {
                    'type': 'BULLISH',
                    'strength': 0.6
                }
            }

        return None


class AdvancedIndicatorFusion:
    """高级指标融合系统 - 使用置信度加权、信号一致性和动态自适应权重"""

    def __init__(self):
        # 指标置信度基准（基于历史表现和理论可靠性）
        self.base_confidence = {
            'EMA_ALIGNMENT': 0.85,  # EMA排列非常可靠
            'MACD_TREND': 0.75,  # MACD趋势较可靠
            'MACD_HIST': 0.70,  # MACD柱状图
            'ADX': 0.80,  # ADX趋势强度
            'PLUS_DI': 0.70,  # DI指标
            'MINUS_DI': 0.70,
            'RSI': 0.65,  # RSI动量
            'STOCH': 0.60,  # 随机指标
            'KDJ': 0.65,  # KDJ指标
            'BOLLINGER': 0.60,  # 布林带
            'ATR': 0.55,  # ATR波动率
            'PRICE_ACTION': 0.70,  # 价格行为
            'PATTERN': 0.75,  # 技术形态
            'VOLUME': 0.55  # 成交量
        }

        # 指标历史表现跟踪（用于动态调整置信度）
        self.performance_history = {key: deque(maxlen=50) for key in self.base_confidence.keys()}

    def calculate_indicator_confidence(self, indicator_name: str, indicator_value: float,
                                       market_state: str, strength: float) -> float:
        """
        计算单个指标的置信度

        Args:
            indicator_name: 指标名称
            indicator_value: 指标值（归一化到-1到1）
            market_state: 市场状态
            strength: 指标强度（0-1）

        Returns:
            置信度分数（0-1）
        """
        base_conf = self.base_confidence.get(indicator_name, 0.5)

        # 根据指标强度调整置信度（强度越高，置信度越高）
        strength_factor = 0.5 + strength * 0.5  # 0.5-1.0

        # 根据市场状态调整置信度
        state_factor = 1.0
        if indicator_name in ['EMA_ALIGNMENT', 'MACD_TREND', 'ADX']:
            # 趋势指标在趋势市更可靠
            if market_state == 'TRENDING':
                state_factor = 1.2
            elif market_state == 'RANGING':
                state_factor = 0.8
        elif indicator_name in ['RSI', 'STOCH', 'KDJ']:
            # 震荡指标在震荡市更可靠
            if market_state == 'RANGING':
                state_factor = 1.2
            elif market_state == 'TRENDING':
                state_factor = 0.9

        # 根据指标值的极端程度调整（极端值更可靠）
        extreme_factor = 1.0
        abs_value = abs(indicator_value)
        if abs_value > 0.7:  # 极端值
            extreme_factor = 1.15
        elif abs_value < 0.2:  # 中性值
            extreme_factor = 0.85

        confidence = base_conf * strength_factor * min(state_factor, 1.2) * extreme_factor
        return min(1.0, max(0.0, confidence))

    def calculate_signal_consistency(self, indicator_signals: Dict[str, float]) -> float:
        """
        计算信号一致性评分

        Args:
            indicator_signals: 指标信号字典 {指标名: 信号值(-1到1)}

        Returns:
            一致性分数（0-1），1表示完全一致
        """
        if not indicator_signals:
            return 0.0

        signals = list(indicator_signals.values())
        if not signals:
            return 0.0

        # 计算信号的方向一致性
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        total_signals = len(signals)

        if total_signals == 0:
            return 0.0

        # 一致性 = 同向信号占比
        consistency = max(positive_signals, negative_signals) / total_signals

        # 如果信号强度都很高，额外加分
        avg_strength = np.mean([abs(s) for s in signals])
        if avg_strength > 0.6:
            consistency = min(1.0, consistency * 1.1)

        return consistency

    def fuse_indicators_advanced(self, indicator_data: Dict[str, Any],
                                 market_state: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        高级指标融合：使用置信度加权、一致性和动态权重

        Args:
            indicator_data: 指标数据字典
            market_state: 市场状态
            weights: 基础权重系统

        Returns:
            融合结果 {'direction': -1/0/1, 'confidence': 0-1, 'consistency': 0-1, 'final_score': 0-1}
        """
        # 提取指标信号
        indicator_signals = {}
        indicator_confidences = {}
        indicator_strengths = {}

        # EMA信号
        ema_alignment = indicator_data.get('EMA_ALIGNMENT', 0)
        if abs(ema_alignment) > 0.1:
            indicator_signals['EMA_ALIGNMENT'] = np.sign(ema_alignment) * min(1.0, abs(ema_alignment))
            indicator_strengths['EMA_ALIGNMENT'] = abs(ema_alignment)
            indicator_confidences['EMA_ALIGNMENT'] = self.calculate_indicator_confidence(
                'EMA_ALIGNMENT', ema_alignment, market_state, abs(ema_alignment)
            )

        # MACD信号
        macd_trend = indicator_data.get('MACD_TREND', 0)
        if abs(macd_trend) > 0.1:
            indicator_signals['MACD_TREND'] = np.sign(macd_trend) * min(1.0, abs(macd_trend))
            indicator_strengths['MACD_TREND'] = abs(macd_trend)
            indicator_confidences['MACD_TREND'] = self.calculate_indicator_confidence(
                'MACD_TREND', macd_trend, market_state, abs(macd_trend)
            )

        # ADX信号（只用于确认趋势强度，不直接产生方向）
        adx = indicator_data.get('ADX', 0)
        adx_strength = min(1.0, (adx - 20) / 40) if adx > 20 else 0.0

        # DI信号
        plus_di = indicator_data.get('PLUS_DI', 0)
        minus_di = indicator_data.get('MINUS_DI', 0)
        if plus_di > minus_di and plus_di > 20:
            di_signal = min(1.0, (plus_di - minus_di) / 40)
            indicator_signals['DI'] = di_signal
            indicator_strengths['DI'] = di_signal
            indicator_confidences['DI'] = self.calculate_indicator_confidence(
                'PLUS_DI', di_signal, market_state, di_signal
            )
        elif minus_di > plus_di and minus_di > 20:
            di_signal = -min(1.0, (minus_di - plus_di) / 40)
            indicator_signals['DI'] = di_signal
            indicator_strengths['DI'] = abs(di_signal)
            indicator_confidences['DI'] = self.calculate_indicator_confidence(
                'MINUS_DI', abs(di_signal), market_state, abs(di_signal)
            )

        # RSI信号
        rsi = indicator_data.get('RSI_14', 50)
        if rsi < 35:
            rsi_signal = min(1.0, (35 - rsi) / 15)  # 超卖，买入信号
            indicator_signals['RSI'] = rsi_signal
            indicator_strengths['RSI'] = rsi_signal
            indicator_confidences['RSI'] = self.calculate_indicator_confidence(
                'RSI', rsi_signal, market_state, rsi_signal
            )
        elif rsi > 65:
            rsi_signal = -min(1.0, (rsi - 65) / 15)  # 超买，卖出信号
            indicator_signals['RSI'] = rsi_signal
            indicator_strengths['RSI'] = abs(rsi_signal)
            indicator_confidences['RSI'] = self.calculate_indicator_confidence(
                'RSI', abs(rsi_signal), market_state, abs(rsi_signal)
            )

        # KDJ信号
        kdj_k = indicator_data.get('KDJ_K', 50)
        kdj_d = indicator_data.get('KDJ_D', 50)
        if indicator_data.get('KDJ_GOLDEN_CROSS', False):
            kdj_signal = min(1.0, abs(kdj_k - kdj_d) / 20)
            indicator_signals['KDJ'] = kdj_signal
            indicator_strengths['KDJ'] = kdj_signal
            indicator_confidences['KDJ'] = self.calculate_indicator_confidence(
                'KDJ', kdj_signal, market_state, kdj_signal
            )
        elif indicator_data.get('KDJ_DEATH_CROSS', False):
            kdj_signal = -min(1.0, abs(kdj_k - kdj_d) / 20)
            indicator_signals['KDJ'] = kdj_signal
            indicator_strengths['KDJ'] = abs(kdj_signal)
            indicator_confidences['KDJ'] = self.calculate_indicator_confidence(
                'KDJ', abs(kdj_signal), market_state, abs(kdj_signal)
            )

        if not indicator_signals:
            return {'direction': 0, 'confidence': 0.0, 'consistency': 0.0, 'final_score': 0.0}

        # 计算信号一致性
        consistency = self.calculate_signal_consistency(indicator_signals)

        # 置信度加权融合
        weighted_sum = 0.0
        total_weight = 0.0

        for indicator_name, signal_value in indicator_signals.items():
            confidence = indicator_confidences.get(indicator_name, 0.5)
            strength = indicator_strengths.get(indicator_name, 0.5)

            # 动态权重 = 基础权重 * 置信度 * 强度
            dynamic_weight = confidence * strength

            weighted_sum += signal_value * dynamic_weight
            total_weight += dynamic_weight

        if total_weight == 0:
            return {'direction': 0, 'confidence': 0.0, 'consistency': 0.0, 'final_score': 0.0}

        # 归一化融合信号
        fused_signal = weighted_sum / total_weight

        # 计算最终置信度（考虑一致性和平均置信度）
        avg_confidence = np.mean(list(indicator_confidences.values()))
        final_confidence = avg_confidence * (0.6 + consistency * 0.4)  # 一致性占40%权重

        # 应用一致性加成（一致性越高，信号越强）
        consistency_boost = 1.0 + (consistency - 0.5) * 0.4  # 一致性>0.5时增强，<0.5时减弱
        final_score = abs(fused_signal) * final_confidence * consistency_boost

        # 确定方向
        direction = 1 if fused_signal > 0.1 else (-1 if fused_signal < -0.1 else 0)

        return {
            'direction': direction,
            'confidence': final_confidence,
            'consistency': consistency,
            'final_score': min(1.0, final_score),
            'fused_signal': fused_signal,
            'indicator_count': len(indicator_signals)
        }


class MLSignalEvaluator:
    """机器学习信号评估器 - 评估信号质量和预测成功率"""

    def __init__(self, model_path: str = "signal_evaluator_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # 特征列表（用于模型训练和预测）
        self.feature_names = [
            'signal_strength', 'fusion_confidence', 'consistency',
            'adx', 'ema_alignment', 'macd_trend', 'rsi_14',
            'stoch_k', 'stoch_d', 'atr_percent', 'bb_position',
            'price_momentum', 'volume_ratio', 'market_state_confidence',
            'trend_start', 'reversal_signal'
        ]

        # 历史信号数据（用于训练）
        self.training_data = []
        self.training_labels = []

        # 评估指标
        self.evaluation_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'training_samples': 0,
            'last_training_time': None
        }

        # 尝试加载已训练的模型
        self._load_model()

    def _load_model(self):
        """加载已训练的模型"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.is_trained = model_data.get('is_trained', False)
                    self.evaluation_metrics = model_data.get('metrics', self.evaluation_metrics)
                    logger.info(f"✅ 成功加载机器学习模型: {self.model_path}")
                    logger.info(f"   模型准确率: {self.evaluation_metrics.get('accuracy', 0):.2%}")
                    logger.info(f"   训练样本数: {self.evaluation_metrics.get('training_samples', 0)}")
            else:
                logger.info("📊 未找到已训练模型，将使用默认评估（需要收集数据后训练）")
        except Exception as e:
            logger.warning(f"⚠️ 加载模型失败: {str(e)}，将使用默认评估")

    def _save_model(self):
        """保存训练好的模型"""
        try:
            if self.model is not None:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'is_trained': self.is_trained,
                    'metrics': self.evaluation_metrics,
                    'feature_names': self.feature_names
                }
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info(f"✅ 模型已保存: {self.model_path}")
        except Exception as e:
            logger.warning(f"⚠️ 保存模型失败: {str(e)}")

    def extract_features(self, signal: Dict, indicators: Dict, market_state: str, state_confidence: float,
                         data_engine=None) -> np.ndarray:
        """从信号和指标中提取特征向量"""
        try:
            if data_engine:
                self.data_engine = data_engine

            features = []

            # 信号特征
            features.append(signal.get('strength', 0.0))
            features.append(signal.get('fusion_confidence', 0.0))
            features.append(signal.get('consistency', 0.0))

            # 技术指标特征
            features.append(indicators.get('ADX', 0.0))
            features.append(indicators.get('EMA_ALIGNMENT', 0.0))
            features.append(indicators.get('MACD_TREND', 0.0))
            features.append(indicators.get('RSI_14', 50.0))
            features.append(indicators.get('STOCH_K', 50.0))
            features.append(indicators.get('STOCH_D', 50.0))
            features.append(indicators.get('ATR_PERCENT', 0.0))
            features.append(indicators.get('BB_POSITION', 0.0))

            # 价格动量
            prices = list(self.data_engine.price_buffer) if hasattr(self, 'data_engine') and self.data_engine else []
            if len(prices) >= 5:
                price_momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0.0
            else:
                price_momentum = 0.0
            features.append(price_momentum)

            # 成交量比率（如果有）
            features.append(indicators.get('VOLUME_RATIO', 1.0))

            # 市场状态特征
            features.append(state_confidence)
            features.append(1.0 if signal.get('trend_start', False) else 0.0)
            features.append(1.0 if signal.get('reversal_signal', False) else 0.0)

            return np.array(features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"提取特征异常: {str(e)}")
            return np.zeros(len(self.feature_names), dtype=np.float32)

    def evaluate_signal(self, signal: Dict, indicators: Dict, market_state: str,
                        state_confidence: float, data_engine=None) -> Dict[str, Any]:
        """
        评估信号质量

        Returns:
            {
                'quality_score': 0-1, 信号质量评分
                'success_probability': 0-1, 预测成功率
                'recommendation': 'STRONG_BUY'/'BUY'/'NEUTRAL'/'SELL'/'STRONG_SELL',
                'confidence': 0-1, 评估置信度
                'features': dict, 特征值
                'ml_prediction': bool, ML模型预测（如果可用）
                'evaluation_details': dict, 详细评估信息
            }
        """
        try:
            if data_engine:
                self.data_engine = data_engine

            # 提取特征
            features = self.extract_features(signal, indicators, market_state, state_confidence, data_engine)

            # 基础评估（不使用ML）
            base_score = signal.get('strength', 0.0)
            fusion_confidence = signal.get('fusion_confidence', 0.0)
            consistency = signal.get('consistency', 0.0)

            # 基础质量评分 - 优化权重分配
            # 如果fusion_confidence和consistency都为0，主要依赖base_score
            if fusion_confidence == 0 and consistency == 0:
                quality_score = base_score * 0.8  # 主要依赖信号强度
            else:
                quality_score = base_score * 0.35 + fusion_confidence * 0.35 + consistency * 0.30

            # 根据市场状态调整基础评分
            if market_state == 'TRENDING':
                # 趋势市：信号强度更重要
                quality_score = base_score * 0.40 + fusion_confidence * 0.35 + consistency * 0.25
            elif market_state == 'RANGING':
                # 震荡市：一致性更重要
                quality_score = base_score * 0.30 + fusion_confidence * 0.30 + consistency * 0.40

            # ML预测（如果模型已训练）
            ml_prediction = None
            ml_confidence = 0.0
            if self.is_trained and self.model is not None:
                try:
                    # 标准化特征
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
                    # 预测
                    ml_prediction = self.model.predict(features_scaled)[0]
                    # 获取预测概率
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(features_scaled)[0]
                        ml_confidence = max(proba)  # 最高类别概率
                    else:
                        ml_confidence = 0.7  # 默认置信度
                except Exception as e:
                    logger.warning(f"ML预测异常: {str(e)}")

            # 综合评估
            if ml_prediction is not None:
                # 使用ML预测调整质量评分
                ml_weight = 0.4
                base_weight = 0.6
                quality_score = base_score * base_weight + ml_prediction * ml_weight * ml_confidence
                success_probability = ml_prediction * ml_confidence
            else:
                # 仅使用基础评估 - 优化成功率估算
                # 根据信号强度和市场状态调整成功率估算
                if market_state == 'TRENDING':
                    success_probability = quality_score * 0.85  # 趋势市：成功率估算更高
                elif market_state == 'RANGING':
                    success_probability = quality_score * 0.75  # 震荡市：成功率估算中等
                else:
                    success_probability = quality_score * 0.80  # 其他：标准估算

                # 如果信号强度很高，进一步提高成功率估算
                if base_score >= 0.6:
                    success_probability = min(1.0, success_probability * 1.15)
                elif base_score >= 0.5:
                    success_probability = min(1.0, success_probability * 1.10)

            # 生成推荐
            if quality_score >= 0.75:
                recommendation = 'STRONG_BUY' if signal.get('direction') == 'BUY' else 'STRONG_SELL'
            elif quality_score >= 0.6:
                recommendation = 'BUY' if signal.get('direction') == 'BUY' else 'SELL'
            elif quality_score >= 0.4:
                recommendation = 'NEUTRAL'
            else:
                recommendation = 'SELL' if signal.get('direction') == 'BUY' else 'BUY'

            # 特征字典
            feature_dict = {name: float(features[i]) for i, name in enumerate(self.feature_names)}

            # 详细评估信息
            evaluation_details = {
                'base_score': base_score,
                'fusion_confidence': fusion_confidence,
                'consistency': consistency,
                'ml_prediction': float(ml_prediction) if ml_prediction is not None else None,
                'ml_confidence': ml_confidence,
                'adx': indicators.get('ADX', 0),
                'ema_alignment': indicators.get('EMA_ALIGNMENT', 0),
                'market_state': market_state,
                'state_confidence': state_confidence
            }

            return {
                'quality_score': quality_score,
                'success_probability': success_probability,
                'recommendation': recommendation,
                'confidence': ml_confidence if ml_prediction is not None else 0.6,
                'features': feature_dict,
                'ml_prediction': ml_prediction is not None,
                'evaluation_details': evaluation_details
            }
        except Exception as e:
            logger.warning(f"信号评估异常: {str(e)}")
            return {
                'quality_score': 0.5,
                'success_probability': 0.5,
                'recommendation': 'NEUTRAL',
                'confidence': 0.5,
                'features': {},
                'ml_prediction': False,
                'evaluation_details': {}
            }

    def record_signal_outcome(self, signal_features: np.ndarray, was_profitable: bool,
                              profit_usd: float, hold_duration: float):
        """记录信号结果（用于后续训练）"""
        try:
            # 标签：盈利>5美元且持仓时间>30秒为成功（1），否则为失败（0）
            label = 1 if (was_profitable and profit_usd > 5.0 and hold_duration > 30) else 0

            self.training_data.append(signal_features)
            self.training_labels.append(label)

            # 限制训练数据大小（最多保留10000条）
            if len(self.training_data) > 10000:
                self.training_data = self.training_data[-10000:]
                self.training_labels = self.training_labels[-10000:]
        except Exception as e:
            logger.warning(f"记录信号结果异常: {str(e)}")

    def train_model(self, min_samples: int = 100):
        """训练机器学习模型"""
        try:
            if len(self.training_data) < min_samples:
                logger.info(f"⏸️ 训练样本不足: {len(self.training_data)} < {min_samples}，暂不训练")
                return False

            logger.info(f"🤖 开始训练ML模型，样本数: {len(self.training_data)}")

            # 转换为numpy数组
            X = np.array(self.training_data)
            y = np.array(self.training_labels)

            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 训练模型（使用梯度提升，通常比随机森林更适合小样本）
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)

            # 评估模型
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # 更新评估指标
            self.evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_samples': len(self.training_data),
                'last_training_time': datetime.now().isoformat()
            }

            self.is_trained = True

            # 保存模型
            self._save_model()

            # 输出详细报告
            logger.info(f"✅ 模型训练完成！")
            logger.info(f"   准确率: {accuracy:.2%}")
            logger.info(f"   精确率: {precision:.2%}")
            logger.info(f"   召回率: {recall:.2%}")
            logger.info(f"   F1分数: {f1:.2%}")
            logger.info(f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

            # 输出分类报告
            report = classification_report(y_test, y_pred, target_names=['失败', '成功'], zero_division=0)
            logger.info(f"\n分类报告:\n{report}")

            return True
        except Exception as e:
            logger.error(f"训练模型异常: {str(e)}")
            traceback.print_exc()
            return False

    def get_evaluation_report(self) -> Dict[str, Any]:
        """获取评估报告"""
        return {
            'model_status': 'trained' if self.is_trained else 'untrained',
            'metrics': self.evaluation_metrics.copy(),
            'training_samples': len(self.training_data),
            'feature_count': len(self.feature_names),
            'model_path': self.model_path
        }


class ProfessionalSignalGenerator:
    """专业信号生成器 - 基于市场状态和多重指标"""

    def __init__(self, data_engine: ProfessionalTickDataEngine, market_analyzer: AdvancedMarketStateAnalyzer):

        self.data_engine = data_engine

        self.market_analyzer = market_analyzer

        self.last_signal_time = 0

        self.signal_history = deque(maxlen=100)

        self.confirmation_count = 0

        # 初始化技术形态识别器
        self.pattern_recognizer = TechnicalPatternRecognizer(data_engine)

        # 初始化高级指标融合系统
        self.indicator_fusion = AdvancedIndicatorFusion()

        # 初始化机器学习信号评估器
        self.ml_evaluator = MLSignalEvaluator()
        self.ml_evaluator.data_engine = data_engine

    def generate_trading_signal(self) -> Optional[Dict[str, Any]]:

        """生成交易信号"""

        if not self.data_engine.initialized:
            return None

        try:

            # 检查信号间隔

            current_time = time.time()
            min_interval = ProfessionalComplexConfig.SIGNAL_GENERATION['FILTERS']['MIN_TICKS_BETWEEN_SIGNALS']

            if current_time - self.last_signal_time < min_interval:
                # 静默返回，避免日志过多
                return None

            # 获取市场状态

            market_state, state_confidence = self.market_analyzer.analyze_complex_market_state()

            # 降低置信度阈值，因为归一化后概率可能较低
            # 根据市场状态动态调整置信度阈值
            if market_state == 'TRENDING':
                confidence_threshold = 0.25  # 趋势市：0.25
            elif market_state == 'RANGING':
                confidence_threshold = 0.20  # 震荡市：0.20（更宽松）
            elif market_state == 'VOLATILE':
                confidence_threshold = 0.25  # 波动市：0.25
            else:
                confidence_threshold = 0.25

            if state_confidence < confidence_threshold:
                # 记录为什么没有生成信号（降低频率）
                if int(current_time) % 60 == 0:  # 每60秒记录一次
                    logger.info(
                        f"⏸️ 市场状态置信度不足: {market_state} (置信度: {state_confidence:.2f} < {confidence_threshold:.2f})，跳过信号生成")
                return None

            # 获取技术指标

            indicators = self.data_engine.calculate_complex_indicators()

            if not indicators:
                # 记录为什么没有生成信号（降低频率）
                if int(current_time) % 60 == 0:  # 每60秒记录一次
                    logger.warning(f"⚠️ 无法计算技术指标，跳过信号生成")
                return None

            # 获取当前价格和点差

            current_tick = self.data_engine.tick_buffer[-1] if self.data_engine.tick_buffer else None

            if not current_tick:
                return None

            current_price = indicators.get('CURRENT_PRICE', current_tick['mid_price'])

            spread = current_tick['spread']

            # 移除点差过滤，点差信息仅用于记录

            # 检查EMA趋势：允许轻微不明确时仍可交易（但降低仓位）
            ema_trend = indicators.get('EMA_TREND', 'UNCERTAIN')
            ema_alignment = indicators.get('EMA_ALIGNMENT', 0)

            # 标记是否为弱趋势（用于后续信号处理）
            is_weak_trend = False

            # 如果EMA趋势不明确，但EMA对齐度有一定方向性（>0.3或<-0.3），允许交易
            if ema_trend == 'UNCERTAIN':
                if abs(ema_alignment) > 0.3:
                    # EMA有轻微方向性，允许交易但标记为弱趋势
                    logger.debug(f"📊 EMA趋势不明确但有一定方向性（对齐度={ema_alignment:.2f}），允许交易")
                    is_weak_trend = True
                else:
                    # 完全震荡市，不生成交易信号
                    if int(current_time) % 60 == 0:  # 每60秒记录一次
                        logger.info(f"⏸️ EMA趋势为完全震荡市（对齐度={ema_alignment:.2f}），不进行交易")
                    return None

            # 根据市场状态生成信号

            signal = None

            if market_state == 'TRENDING':

                signal = self._generate_trending_signal(indicators, current_price, spread)

            elif market_state == 'RANGING':

                signal = self._generate_ranging_signal(indicators, current_price, spread)

            elif market_state == 'VOLATILE':

                signal = self._generate_volatile_signal(indicators, current_price, spread)

            if signal:
                # 如果之前标记为弱趋势，添加到信号中
                if is_weak_trend:
                    signal['weak_trend'] = True

                if signal['strength'] >= ProfessionalComplexConfig.SIGNAL_GENERATION['MIN_STRENGTH']:
                    # 使用ML评估信号质量
                    ml_evaluation = self.ml_evaluator.evaluate_signal(
                        signal, indicators, market_state, state_confidence, self.data_engine
                    )

                    # 将ML评估结果添加到信号中
                    signal['ml_evaluation'] = ml_evaluation
                    signal['quality_score'] = ml_evaluation['quality_score']
                    signal['success_probability'] = ml_evaluation['success_probability']
                    signal['recommendation'] = ml_evaluation['recommendation']

                    # 动态质量评分阈值：根据市场状态和ML模型状态调整
                    # 如果ML模型未训练，使用更宽松的标准
                    if not self.ml_evaluator.is_trained:
                        # ML模型未训练，使用基础评估，降低阈值
                        if market_state == 'TRENDING':
                            min_quality_score = 0.45  # 趋势市：0.45
                        elif market_state == 'RANGING':
                            min_quality_score = 0.35  # 震荡市：0.35
                        elif market_state == 'VOLATILE':
                            min_quality_score = 0.40  # 波动市：0.40
                        else:
                            min_quality_score = 0.40

                        # 如果信号强度很高，进一步降低要求
                        if signal['strength'] >= 0.6:
                            min_quality_score *= 0.85  # 降低15%
                        elif signal['strength'] >= 0.5:
                            min_quality_score *= 0.90  # 降低10%
                    else:
                        # ML模型已训练，使用更严格的标准
                        if market_state == 'TRENDING':
                            min_quality_score = 0.55  # 趋势市：0.55
                        elif market_state == 'RANGING':
                            min_quality_score = 0.45  # 震荡市：0.45
                        elif market_state == 'VOLATILE':
                            min_quality_score = 0.50  # 波动市：0.50
                        else:
                            min_quality_score = 0.50

                    # 对于弱趋势信号，进一步降低要求
                    if signal.get('weak_trend', False):
                        min_quality_score *= 0.85  # 降低15%

                    if ml_evaluation['quality_score'] < min_quality_score:
                        if int(current_time) % 60 == 0:
                            logger.info(f"⏸️ 信号质量不足: {signal.get('direction', 'UNKNOWN')} "
                                        f"质量评分: {ml_evaluation['quality_score']:.2f} < {min_quality_score:.2f}, "
                                        f"成功率: {ml_evaluation['success_probability']:.2%}, "
                                        f"市场状态: {market_state}, ML训练: {self.ml_evaluator.is_trained}")
                        return None

                    # 对于反转信号，需要更高的质量评分
                    if signal.get('reversal_signal', False):
                        min_reversal_quality = 0.75
                        if ml_evaluation['quality_score'] < min_reversal_quality:
                            if int(current_time) % 60 == 0:
                                logger.info(
                                    f"⏸️ 反转信号质量不足: 质量评分: {ml_evaluation['quality_score']:.2f} < {min_reversal_quality}")
                            return None

                    # 对于趋势启动信号，需要较高的成功率
                    if signal.get('trend_start', False):
                        min_trend_start_prob = 0.65
                        if ml_evaluation['success_probability'] < min_trend_start_prob:
                            if int(current_time) % 60 == 0:
                                logger.info(
                                    f"⏸️ 趋势启动信号成功率不足: {ml_evaluation['success_probability']:.2%} < {min_trend_start_prob:.2%}")
                            return None

                    signal['market_state'] = market_state
                    signal['state_confidence'] = state_confidence
                    signal['timestamp'] = current_time
                    self.last_signal_time = current_time
                    self.signal_history.append(signal)

                    # 详细日志
                    logger.info(f"📈 生成高质量信号: {signal['direction']} "
                                f"强度: {signal['strength']:.2f} "
                                f"质量: {ml_evaluation['quality_score']:.2f} "
                                f"成功率: {ml_evaluation['success_probability']:.2%} "
                                f"推荐: {ml_evaluation['recommendation']} "
                                f"价格: {current_price:.2f}")
                    if ml_evaluation['ml_prediction']:
                        logger.info(f"   🤖 ML预测: 置信度={ml_evaluation['confidence']:.2%}")

                    return signal
                else:
                    # 记录信号强度不足（降低频率）
                    if int(current_time) % 60 == 0:  # 每60秒记录一次
                        logger.info(f"⏸️ 信号强度不足: {signal.get('direction', 'UNKNOWN')} "
                                    f"强度: {signal['strength']:.2f} < {ProfessionalComplexConfig.SIGNAL_GENERATION['MIN_STRENGTH']}")
            else:
                # 记录为什么没有生成信号（降低频率，添加详细诊断信息）
                if int(current_time) % 60 == 0:  # 每60秒记录一次
                    # 添加详细的诊断信息
                    try:
                        ema_alignment = indicators.get('EMA_ALIGNMENT', 'N/A')
                        macd_trend = indicators.get('MACD_TREND', 'N/A')
                        adx = indicators.get('ADX', 'N/A')
                        rsi_14 = indicators.get('RSI_14', 'N/A')
                        atr_percent = indicators.get('ATR_PERCENT', 'N/A')

                        if market_state == 'TRENDING':
                            logger.info(f"⏸️ TRENDING状态未生成信号 - 诊断: EMA={ema_alignment}, "
                                        f"MACD={macd_trend}, ADX={adx}, RSI={rsi_14}, ATR%={atr_percent}")
                        elif market_state == 'RANGING':
                            stoch_k = indicators.get('STOCH_K', 'N/A')
                            bb_position = indicators.get('BB_POSITION', 'N/A')
                            logger.info(f"⏸️ RANGING状态未生成信号 - 诊断: RSI={rsi_14}, "
                                        f"StochK={stoch_k}, BB位置={bb_position}")
                        elif market_state == 'VOLATILE':
                            bb_upper = indicators.get('BB_UPPER_2.0', 'N/A')
                            bb_lower = indicators.get('BB_LOWER_2.0', 'N/A')
                            logger.info(f"⏸️ VOLATILE状态未生成信号 - 诊断: 价格={current_price:.2f}, "
                                        f"BB上轨={bb_upper}, BB下轨={bb_lower}, ATR%={atr_percent}")
                        else:
                            logger.info(f"⏸️ 市场状态 {market_state} 下未生成信号（可能条件不满足）")
                    except:
                        logger.info(f"⏸️ 市场状态 {market_state} 下未生成信号（可能条件不满足）")

            return None

        except Exception as e:

            logger.error(f"生成信号异常: {str(e)}")

            return None

    def _detect_trend_start(self, indicators: Dict, current_price: float) -> Optional[Dict]:
        """检测趋势启动信号 - 捕捉早期趋势"""

        if not ProfessionalComplexConfig.TREND_START_DETECTION['ENABLE']:
            return None

        try:
            # 获取历史ADX值
            current_adx = indicators.get('ADX', 0)
            previous_adx = indicators.get('ADX_PREV', 0)

            # 趋势启动信号：ADX从低（<15）快速上升（>18）
            adx_rising = (current_adx > ProfessionalComplexConfig.TREND_START_DETECTION['ADX_RISING_THRESHOLD'] and
                          previous_adx < ProfessionalComplexConfig.TREND_START_DETECTION['ADX_PREV_THRESHOLD'] and
                          current_adx > previous_adx * 1.2)

            # EMA突破信号
            ema_5 = indicators.get('EMA_5', 0)
            ema_15 = indicators.get('EMA_15', 0)
            ema_30 = indicators.get('EMA_30', 0)
            ema_alignment = indicators.get('EMA_ALIGNMENT', 0)

            # 多头启动：价格突破EMA15，且EMA5上穿EMA15
            bullish_breakout = (current_price > ema_15 and
                                ema_5 > ema_15 and
                                ema_alignment > 0.2)

            # 空头启动：价格跌破EMA15，且EMA5下穿EMA15
            bearish_breakout = (current_price < ema_15 and
                                ema_5 < ema_15 and
                                ema_alignment < -0.2)

            # MACD金叉/死叉确认
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_SIGNAL', 0)
            macd_hist = indicators.get('MACD_HIST', 0)

            bullish_cross = macd > macd_signal and macd_hist > 0
            bearish_cross = macd < macd_signal and macd_hist < 0

            # 价格动量加速
            prices = list(self.data_engine.price_buffer)
            momentum_acceleration = False
            if len(prices) >= 10:
                momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
                momentum_10 = (prices[-5] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
                momentum_acceleration = abs(momentum_5) > abs(momentum_10) * \
                                        ProfessionalComplexConfig.TREND_START_DETECTION['MOMENTUM_ACCELERATION']

            # 综合判断 - 需要更多确认条件以提高准确性
            # 1. 必须同时满足ADX上升和动量加速（更严格的条件）
            strong_trend_start = adx_rising and momentum_acceleration

            # 2. 或者ADX上升且EMA突破且MACD确认（中等条件）
            moderate_trend_start = adx_rising and (bullish_breakout or bearish_breakout) and (
                        bullish_cross or bearish_cross)

            # 3. 或者动量加速且EMA突破且MACD确认（早期捕捉）
            early_trend_start = momentum_acceleration and (bullish_breakout or bearish_breakout) and (
                        bullish_cross or bearish_cross)

            if strong_trend_start or moderate_trend_start or early_trend_start:
                # 计算基础信号强度
                base_strength = 0.4 + (current_adx / 100) * 0.3  # 0.4-0.7

                # 根据确认条件数量增强信号
                confirmation_bonus = 0.0
                if adx_rising:
                    confirmation_bonus += 0.1
                if momentum_acceleration:
                    confirmation_bonus += 0.1
                if bullish_breakout or bearish_breakout:
                    confirmation_bonus += 0.05
                if bullish_cross or bearish_cross:
                    confirmation_bonus += 0.05

                # 强趋势启动信号额外加成
                if strong_trend_start:
                    confirmation_bonus += 0.1

                signal_strength = base_strength + confirmation_bonus

                # 添加RSI确认（避免在极端超买/超卖时开仓）
                rsi_14 = indicators.get('RSI_14', 50)
                if bullish_breakout and bullish_cross:
                    # 多头：RSI不应过高（避免追高）
                    if rsi_14 > 75:
                        if int(time.time()) % 60 == 0:
                            logger.info(f"⏸️ 趋势启动信号被过滤: RSI过高 ({rsi_14:.1f})")
                        return None
                    # RSI在合理区间时增强信号
                    if 30 < rsi_14 < 70:
                        signal_strength += 0.05

                    if signal_strength >= ProfessionalComplexConfig.TREND_START_DETECTION['MIN_SIGNAL_STRENGTH']:
                        return {
                            'direction': 'BUY',
                            'strength': min(1.0, signal_strength),
                            'entry_price': current_price,
                            'trend_start': True,
                            'adx_rising': adx_rising,
                            'momentum_acceleration': momentum_acceleration,
                            'fusion_confidence': min(1.0, signal_strength * 0.9),
                            'consistency': 0.8 if strong_trend_start else 0.7
                        }
                elif bearish_breakout and bearish_cross:
                    # 空头：RSI不应过低（避免追低）
                    if rsi_14 < 25:
                        if int(time.time()) % 60 == 0:
                            logger.info(f"⏸️ 趋势启动信号被过滤: RSI过低 ({rsi_14:.1f})")
                        return None
                    # RSI在合理区间时增强信号
                    if 30 < rsi_14 < 70:
                        signal_strength += 0.05

                    if signal_strength >= ProfessionalComplexConfig.TREND_START_DETECTION['MIN_SIGNAL_STRENGTH']:
                        return {
                            'direction': 'SELL',
                            'strength': min(1.0, signal_strength),
                            'entry_price': current_price,
                            'trend_start': True,
                            'adx_rising': adx_rising,
                            'momentum_acceleration': momentum_acceleration,
                            'fusion_confidence': min(1.0, signal_strength * 0.9),
                            'consistency': 0.8 if strong_trend_start else 0.7
                        }

        except Exception as e:
            logger.warning(f"趋势启动检测异常: {str(e)}")

            return None

    def _generate_trending_signal(self, indicators: Dict, current_price: float, spread: float) -> Optional[Dict]:

        """生成趋势市信号 - 使用高级指标融合系统"""

        # 首先尝试检测趋势启动信号（早期捕捉）
        if ProfessionalComplexConfig.TREND_START_DETECTION['ENABLE']:
            trend_start_signal = self._detect_trend_start(indicators, current_price)
            if trend_start_signal:
                logger.info(
                    f"🚀 检测到趋势启动信号: {trend_start_signal['direction']} 强度: {trend_start_signal['strength']:.2f}")
                return trend_start_signal

        weights = ProfessionalComplexConfig.SIGNAL_GENERATION['WEIGHT_SYSTEM']['TRENDING']

        # 获取所有指标值
        adx = indicators.get('ADX', 0)

        # ADX基础要求（必须有趋势强度）
        if adx <= 20:
            return None

        # 使用高级融合系统
        fusion_result = self.indicator_fusion.fuse_indicators_advanced(
            indicators, 'TRENDING', weights
        )

        direction = fusion_result['direction']
        fusion_confidence = fusion_result['confidence']
        consistency = fusion_result['consistency']
        final_score = fusion_result['final_score']
        fused_signal = fusion_result['fused_signal']
        indicator_count = fusion_result['indicator_count']

        # 如果融合系统没有确定方向，返回None
        if direction == 0:
            return None

        # 获取其他指标用于增强信号
        ema_alignment = indicators.get('EMA_ALIGNMENT', 0)
        macd_trend = indicators.get('MACD_TREND', 0)
        rsi_14 = indicators.get('RSI_14', 50)
        stoch_k = indicators.get('STOCH_K', 50)
        stoch_d = indicators.get('STOCH_D', 50)
        plus_di = indicators.get('PLUS_DI', 0)
        minus_di = indicators.get('MINUS_DI', 0)
        macd_hist = indicators.get('MACD_HIST', 0)

        # 添加调试日志（每60秒输出一次）
        current_time = time.time()
        if int(current_time) % 60 == 0:
            logger.info(f"🔍 高级融合系统: 方向={direction}, 置信度={fusion_confidence:.2f}, "
                        f"一致性={consistency:.2f}, 最终得分={final_score:.2f}, "
                        f"指标数={indicator_count}, 融合信号={fused_signal:.3f}")

        # 基于融合结果计算信号强度
        signal_score = final_score

        # 基于融合结果增强信号（一致性越高，信号越强）
        # 融合系统已经考虑了大部分指标，这里只添加额外的确认

        # 一致性加成（融合系统已经计算了一致性，这里应用加成）
        if consistency > 0.7:
            signal_score *= 1.20  # 高一致性，增强20%（从15%提高）
        elif consistency > 0.6:
            signal_score *= 1.10  # 中等一致性，增强10%
        elif consistency < 0.5:
            signal_score *= 0.95  # 低一致性，减弱5%（从10%降低）

        # 如果ADX很高，说明趋势很强，增强信号
        adx = indicators.get('ADX', 0)
        if adx > 30:
            signal_score *= 1.10  # ADX>30，增强10%
        elif adx > 25:
            signal_score *= 1.05  # ADX>25，增强5%

        # 价格动量确认
        prices = list(self.data_engine.price_buffer)
        if len(prices) >= 5:
            recent_momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
            # 价格动量与信号方向一致时增强
            if (direction == 1 and recent_momentum > 0) or (direction == -1 and recent_momentum < 0):
                signal_score += weights.get('PRICE_ACTION', 0.10) * 0.5  # 降低权重，因为融合系统已考虑
                if abs(recent_momentum) > 0.001:
                    signal_score += 0.03

        # 技术形态确认
        if len(prices) >= 20:
            highs = list(self.data_engine.high_buffer)
            lows = list(self.data_engine.low_buffer)
            if len(highs) >= 20 and len(lows) >= 20:
                patterns = self.pattern_recognizer.detect_patterns(prices, highs, lows)
                for pattern_name, pattern_data in patterns.items():
                    pattern_type = pattern_data.get('type', 'NEUTRAL')
                    pattern_strength = pattern_data.get('strength', 0.5)
                    # 形态方向与信号方向一致时增强
                    if (direction == 1 and pattern_type == 'BULLISH') or (
                            direction == -1 and pattern_type == 'BEARISH'):
                        pattern_score = pattern_strength * weights.get('PATTERN_RECOGNITION', 0.10) * 0.5
                        signal_score += pattern_score
                        if int(current_time) % 60 == 0:
                            logger.info(
                                f"🔍 技术形态确认: {pattern_name} ({pattern_type}), 强度: {pattern_strength:.2f}")

        # 返回信号
        if signal_score > 0 and direction != 0:
            # 根据指标确认数量进一步增强信号
            if indicator_count >= 5:
                signal_score *= 1.08  # 5个以上指标确认，增强8%
            elif indicator_count >= 4:
                signal_score *= 1.05  # 4个指标确认，增强5%

            if int(current_time) % 60 == 0:
                logger.info(f"📊 高级融合信号: 方向={'BUY' if direction == 1 else 'SELL'}, "
                            f"强度={signal_score:.3f}, 置信度={fusion_confidence:.2f}, "
                            f"一致性={consistency:.2f} (需要≥{ProfessionalComplexConfig.SIGNAL_GENERATION['MIN_STRENGTH']})")

            return {
                'direction': 'BUY' if direction == 1 else 'SELL',
                'strength': min(1.0, signal_score),
                'entry_price': current_price,
                'spread': spread,
                'fusion_confidence': fusion_confidence,
                'consistency': consistency
            }

        return None

    def _generate_ranging_signal(self, indicators: Dict, current_price: float, spread: float) -> Optional[Dict]:

        """生成震荡市信号 - 改进版：使用渐进式评分，捕捉更多反转机会"""

        weights = ProfessionalComplexConfig.SIGNAL_GENERATION['WEIGHT_SYSTEM']['RANGING']

        signal_score = 0.0

        direction = 0

        # 获取所有指标值

        rsi_14 = indicators.get('RSI_14', 50)

        stoch_k = indicators.get('STOCH_K', 50)

        stoch_d = indicators.get('STOCH_D', 50)

        williams = indicators.get('WILLIAMSR', -50)

        cci = indicators.get('CCI', 0)

        bb_position = indicators.get('BB_POSITION', 0.5)

        bb_upper = indicators.get('BB_UPPER_2.0', current_price)

        bb_lower = indicators.get('BB_LOWER_2.0', current_price)

        bb_middle = indicators.get('BB_UPPER_1.0', current_price)  # 使用1.0标准差作为中轨近似

        # KDJ指标
        kdj_k = indicators.get('KDJ_K', 50.0)
        kdj_d = indicators.get('KDJ_D', 50.0)
        kdj_j = indicators.get('KDJ_J', 50.0)
        kdj_golden_cross = indicators.get('KDJ_GOLDEN_CROSS', False)
        kdj_death_cross = indicators.get('KDJ_DEATH_CROSS', False)
        kdj_oversold = indicators.get('KDJ_OVERSOLD', False)
        kdj_overbought = indicators.get('KDJ_OVERBOUGHT', False)

        oscillator_score = 0.0

        bullish_oscillators = 0

        bearish_oscillators = 0

        # 1. RSI分析（渐进式）

        if rsi_14 < 35:  # 放宽超卖条件

            oscillator_score += 0.15

            bullish_oscillators += 1

        if rsi_14 < 30:  # 完全超卖

            oscillator_score += 0.15

            bullish_oscillators += 1

        elif rsi_14 > 65:  # 放宽超买条件

            oscillator_score += 0.15

            bearish_oscillators += 1

        elif rsi_14 > 70:  # 完全超买

            oscillator_score += 0.15

            bearish_oscillators += 1

        # 2. Stochastic分析（渐进式）

        if stoch_k < 25:  # 放宽超卖条件

            oscillator_score += 0.15

            bullish_oscillators += 1

        if stoch_k < 20:  # 完全超卖

            oscillator_score += 0.15

            bullish_oscillators += 1

        elif stoch_k > 75:  # 放宽超买条件

            oscillator_score += 0.15

            bearish_oscillators += 1

        elif stoch_k > 80:  # 完全超买

            oscillator_score += 0.15

            bearish_oscillators += 1

        # 3. Stochastic交叉信号

        if stoch_k > stoch_d and stoch_k < 30:  # 金叉且处于低位

            oscillator_score += 0.1

            bullish_oscillators += 1

        elif stoch_k < stoch_d and stoch_k > 70:  # 死叉且处于高位

            oscillator_score += 0.1

            bearish_oscillators += 1

        # 4. Williams %R分析

        if williams < -75:  # 放宽超卖条件

            oscillator_score += 0.1

            bullish_oscillators += 1

        if williams < -80:  # 完全超卖

            oscillator_score += 0.1

            bullish_oscillators += 1

        elif williams > -25:  # 放宽超买条件

            oscillator_score += 0.1

            bearish_oscillators += 1

        elif williams > -20:  # 完全超买

            oscillator_score += 0.1

            bearish_oscillators += 1

        # 5. CCI分析（新增）

        if cci < -100:  # 超卖

            oscillator_score += 0.1

            bullish_oscillators += 1

        elif cci > 100:  # 超买

            oscillator_score += 0.1

            bearish_oscillators += 1

        # 6. KDJ指标分析
        if kdj_oversold:  # KDJ超卖
            oscillator_score += 0.15
            bullish_oscillators += 1

        if kdj_golden_cross and kdj_k < 30:  # KDJ金叉且在低位
            oscillator_score += 0.15
            bullish_oscillators += 1

        if kdj_overbought:  # KDJ超买
            oscillator_score += 0.15
            bearish_oscillators += 1

        if kdj_death_cross and kdj_k > 70:  # KDJ死叉且在高位
            oscillator_score += 0.15
            bearish_oscillators += 1

        # KDJ在极端区域
        if kdj_k < 20 and kdj_d < 20:
            oscillator_score += 0.1
            bullish_oscillators += 1
        elif kdj_k > 80 and kdj_d > 80:
            oscillator_score += 0.1
            bearish_oscillators += 1

        # 归一化震荡指标分数 - 优化以提高信号强度

        if oscillator_score > 0:
            # 提高震荡指标的权重，并考虑确认数量
            oscillator_weight = weights['OSCILLATORS']  # 0.30
            confirmation_bonus = min(0.15, (bullish_oscillators + bearish_oscillators) * 0.03)  # 每多一个确认指标+3%，最多+15%

            # 如果多个指标一致，增强信号
            consistency_bonus = 0.0
            total_oscillators = bullish_oscillators + bearish_oscillators
            if total_oscillators >= 3:
                consistency_bonus = 0.10  # 3个以上指标一致，+10%
            elif total_oscillators >= 2:
                consistency_bonus = 0.05  # 2个指标一致，+5%

            # 计算基础信号分数
            base_oscillator_score = (oscillator_score / 1.2) * oscillator_weight  # 降低归一化分母，提高分数
            signal_score += base_oscillator_score + confirmation_bonus + consistency_bonus

            # 确定方向

            if bullish_oscillators > bearish_oscillators:

                direction = 1

            elif bearish_oscillators > bullish_oscillators:

                direction = -1

            elif rsi_14 < 50:

                direction = 1

            else:

                direction = -1

        if direction != 0:

            support_resistance_score = 0.0

            if direction == 1:  # 买入信号

                if bb_position < 0.3:  # 放宽条件从0.2到0.3

                    support_resistance_score += 0.5

                if bb_position < 0.2:  # 完全满足

                    support_resistance_score += 0.5

                # 价格接近下轨

                if current_price <= bb_lower * 1.002:
                    support_resistance_score += 0.3

            else:  # 卖出信号

                if bb_position > 0.7:  # 放宽条件从0.8到0.7

                    support_resistance_score += 0.5

                if bb_position > 0.8:  # 完全满足

                    support_resistance_score += 0.5

                # 价格接近上轨

                if current_price >= bb_upper * 0.998:
                    support_resistance_score += 0.3

            if support_resistance_score > 0:
                signal_score += (support_resistance_score / 1.0) * weights['SUPPORT_RESISTANCE']

        if direction != 0:

            prices = list(self.data_engine.price_buffer)

            if len(prices) >= 10:

                # 检查是否在震荡区间

                recent_high = max(prices[-10:])

                recent_low = min(prices[-10:])

                price_range = (recent_high - recent_low) / ((recent_high + recent_low) / 2) if (
                                                                                                           recent_high + recent_low) > 0 else 0

                # 如果价格在区间内震荡，给予模式识别分数

                if price_range < 0.002:  # 低波动，符合震荡市特征

                    signal_score += weights.get('PRICE_PATTERNS', 0.15) * 0.5

                # 检查价格是否在布林带中轨附近（震荡市特征）

                if bb_middle > 0:

                    distance_to_middle = abs(current_price - bb_middle) / bb_middle if bb_middle > 0 else 0

                    if distance_to_middle < 0.001:  # 接近中轨

                        signal_score += weights.get('PRICE_PATTERNS', 0.15) * 0.3

        if direction != 0:
            prices = list(self.data_engine.price_buffer)
            highs = list(self.data_engine.high_buffer)
            lows = list(self.data_engine.low_buffer)

            if len(prices) >= 20 and len(highs) >= 20 and len(lows) >= 20:
                patterns = self.pattern_recognizer.detect_patterns(prices, highs, lows)

                for pattern_name, pattern_data in patterns.items():
                    pattern_type = pattern_data.get('type', 'NEUTRAL')
                    pattern_strength = pattern_data.get('strength', 0.5)

                    # 震荡市特别关注反转形态（双顶/双底、头肩等）
                    if pattern_name in ['DOUBLE_TOP', 'DOUBLE_BOTTOM', 'HEAD_SHOULDER_TOP', 'HEAD_SHOULDER_BOTTOM']:
                        if (direction == 1 and pattern_type == 'BULLISH') or (
                                direction == -1 and pattern_type == 'BEARISH'):
                            pattern_score = pattern_strength * weights.get('PRICE_PATTERNS', 0.15)
                            signal_score += pattern_score
                    # 矩形和旗形也给予分数
                    elif pattern_name in ['RECTANGLE', 'FLAG_PATTERN']:
                        pattern_score = pattern_strength * weights.get('PRICE_PATTERNS', 0.15) * 0.5
                        signal_score += pattern_score

        if signal_score > 0 and direction != 0:
            # 根据确认指标数量增强信号强度
            total_confirmations = bullish_oscillators + bearish_oscillators
            if total_confirmations >= 4:
                signal_score *= 1.15  # 4个以上确认，增强15%
            elif total_confirmations >= 3:
                signal_score *= 1.10  # 3个确认，增强10%
            elif total_confirmations >= 2:
                signal_score *= 1.05  # 2个确认，增强5%

            # 如果RSI或Stochastic在极端区域，额外增强
            if (rsi_14 < 25 or rsi_14 > 75) or (stoch_k < 20 or stoch_k > 80):
                signal_score *= 1.08  # 极端区域，增强8%

            return {

                'direction': 'BUY' if direction == 1 else 'SELL',

                'strength': min(1.0, signal_score),

                'entry_price': current_price,

                'spread': spread,

                'fusion_confidence': min(1.0, signal_score * 0.9),  # 添加融合置信度

                'consistency': min(1.0, total_confirmations / 5.0) if total_confirmations > 0 else 0.5  # 添加一致性评分

            }

        return None

    def _generate_volatile_signal(self, indicators: Dict, current_price: float, spread: float) -> Optional[Dict]:

        """生成高波动市信号 - 改进版：更精准捕捉突破机会"""

        weights = ProfessionalComplexConfig.SIGNAL_GENERATION['WEIGHT_SYSTEM']['VOLATILE']

        signal_score = 0.0

        direction = 0

        # 获取所有指标值

        bb_upper = indicators.get('BB_UPPER_2.0', current_price)

        bb_lower = indicators.get('BB_LOWER_2.0', current_price)

        bb_upper_1 = indicators.get('BB_UPPER_1.0', current_price)

        bb_lower_1 = indicators.get('BB_LOWER_1.0', current_price)

        atr_percent = indicators.get('ATR_PERCENT', 0)

        adx = indicators.get('ADX', 0)

        macd_hist = indicators.get('MACD_HIST', 0)

        prices = list(self.data_engine.price_buffer)

        breakout_score = 0.0

        # 1. 布林带突破（多层级）

        if current_price > bb_upper * 0.998:  # 放宽条件

            breakout_score += 0.4

            direction = 1

        if current_price > bb_upper:  # 完全突破

            breakout_score += 0.6

            direction = 1

        elif current_price < bb_lower * 1.002:  # 放宽条件

            breakout_score += 0.4

            direction = -1

        elif current_price < bb_lower:  # 完全突破

            breakout_score += 0.6

            direction = -1

        # 2. 1.0标准差布林带突破（早期信号）

        if direction == 0:  # 如果2.0标准差未突破，检查1.0标准差

            if current_price > bb_upper_1 * 0.999:

                breakout_score += 0.3

                direction = 1

            elif current_price < bb_lower_1 * 1.001:

                breakout_score += 0.3

                direction = -1

        # 归一化突破分数

        if breakout_score > 0:
            signal_score += (breakout_score / 1.0) * weights['BREAKOUT_SIGNALS']

        if direction != 0 and len(prices) >= 5:

            price_action_score = 0.0

            # 短期动量

            recent_momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0

            # 中期动量（更可靠）

            if len(prices) >= 10:

                medium_momentum = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0

            else:

                medium_momentum = recent_momentum

            if direction == 1:

                # 价格上涨动量确认

                if recent_momentum > 0.0003:  # 降低阈值

                    price_action_score += 0.4

                if recent_momentum > 0.0005:  # 完全满足

                    price_action_score += 0.3

                if medium_momentum > 0.0005:  # 中期动量确认

                    price_action_score += 0.3

            else:  # direction == -1

                # 价格下跌动量确认

                if recent_momentum < -0.0003:  # 降低阈值

                    price_action_score += 0.4

                if recent_momentum < -0.0005:  # 完全满足

                    price_action_score += 0.3

                if medium_momentum < -0.0005:  # 中期动量确认

                    price_action_score += 0.3

            if price_action_score > 0:
                signal_score += (price_action_score / 1.0) * weights['PRICE_ACTION']

        if atr_percent > 0.0006:  # 高波动率确认

            volatility_score = min(1.0, atr_percent / 0.002)  # 归一化

            signal_score += volatility_score * weights['VOLATILITY_INDICATORS']

        if direction != 0 and adx > 25:  # 高波动市也需要趋势强度

            trend_score = min(1.0, adx / 50.0)

            signal_score += trend_score * weights.get('TREND_INDICATORS', 0.15) * 0.5

        if direction != 0:

            if (direction == 1 and macd_hist > 0) or (direction == -1 and macd_hist < 0):
                signal_score += weights.get('MOMENTUM_INDICATORS', 0.05) * 0.5

        if direction != 0:

            volume_profile = self.data_engine.volume_buffer

            if len(volume_profile) >= 5:

                recent_volumes = list(volume_profile)[-5:]

                avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0

                if avg_volume > 0:

                    current_volume = recent_volumes[-1] if recent_volumes else 0

                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

                    # 突破时成交量放大是好的信号

                    if volume_ratio > 1.2:
                        signal_score += 0.05

        if direction != 0:
            prices = list(self.data_engine.price_buffer)
            highs = list(self.data_engine.high_buffer)
            lows = list(self.data_engine.low_buffer)

            if len(prices) >= 20 and len(highs) >= 20 and len(lows) >= 20:
                patterns = self.pattern_recognizer.detect_patterns(prices, highs, lows)

                for pattern_name, pattern_data in patterns.items():
                    pattern_type = pattern_data.get('type', 'NEUTRAL')
                    pattern_strength = pattern_data.get('strength', 0.5)

                    # 高波动市特别关注突破形态
                    if pattern_name in ['RESISTANCE_BREAKOUT', 'SUPPORT_BREAKDOWN', 'ASCENDING_TRIANGLE',
                                        'DESCENDING_TRIANGLE', 'FLAG_PATTERN']:
                        if (direction == 1 and pattern_type == 'BULLISH') or (
                                direction == -1 and pattern_type == 'BEARISH'):
                            pattern_score = pattern_strength * weights.get('BREAKOUT_SIGNALS', 0.20) * 0.5
                            signal_score += pattern_score
                    # 楔形形态也给予分数
                    elif pattern_name in ['RISING_WEDGE', 'FALLING_WEDGE']:
                        if (direction == 1 and pattern_type == 'BULLISH') or (
                                direction == -1 and pattern_type == 'BEARISH'):
                            pattern_score = pattern_strength * weights.get('BREAKOUT_SIGNALS', 0.20) * 0.3
                            signal_score += pattern_score

        if signal_score > 0 and direction != 0:
            return {

                'direction': 'BUY' if direction == 1 else 'SELL',

                'strength': min(1.0, signal_score),

                'entry_price': current_price,

                'spread': spread

            }

        return None


class ComplexRiskManager:
    """复杂风险管理器 - 多层止损和动态仓位"""

    def __init__(self, data_engine: ProfessionalTickDataEngine):

        self.data_engine = data_engine

        self.account_info = None

        self.update_account_info()

    def update_account_info(self):

        """更新账户信息"""

        try:

            account_info = mt5.account_info()

            if account_info:
                self.account_info = {

                    'balance': account_info.balance,

                    'equity': account_info.equity,

                    'margin': account_info.margin,

                    'free_margin': account_info.margin_free,

                    'margin_level': account_info.margin_level if account_info.margin > 0 else 0

                }

        except Exception as e:

            logger.warning(f"更新账户信息异常: {str(e)}")

    def calculate_position_size(self, signal: Dict, entry_price: float,
                                stop_loss: float = None, take_profit: float = None) -> float:

        """计算仓位大小（考虑交易成本和盈亏比）"""

        if not self.account_info:

            self.update_account_info()

            if not self.account_info:
                return ProfessionalComplexConfig.MIN_LOT

        try:

            # 基础风险计算

            balance = self.account_info['balance']

            risk_amount = balance * ProfessionalComplexConfig.RISK_PER_TRADE

            # 计算止损距离

            stop_loss_distance = self.calculate_stop_loss_distance(signal, entry_price)

            if stop_loss_distance <= 0:
                return ProfessionalComplexConfig.MIN_LOT

            # 计算仓位

            symbol_info = self.data_engine.data_validator.symbol_info

            if not symbol_info:
                return ProfessionalComplexConfig.MIN_LOT

            tick_value = symbol_info.trade_tick_value

            if tick_value <= 0:
                tick_value = ProfessionalComplexConfig.POINT_VALUE

            # 获取点差和手续费
            spread = abs(symbol_info.ask - symbol_info.bid) if ProfessionalComplexConfig.SPREAD_COST_ENABLED else 0.0
            commission_per_lot = ProfessionalComplexConfig.COMMISSION_PER_LOT

            # 初步计算手数（不考虑交易成本）
            lot_size = risk_amount / (stop_loss_distance * tick_value)

            # 迭代计算：因为手数影响手续费，需要迭代求解
            for iteration in range(5):  # 最多迭代5次
                # 计算实际风险（包含点差和手续费）
                # 实际风险 = 止损损失 + 点差成本 + 手续费
                total_cost_per_lot = spread + commission_per_lot
                actual_risk_per_lot = stop_loss_distance * tick_value + total_cost_per_lot

                # 重新计算手数
                if actual_risk_per_lot > 0:
                    new_lot_size = risk_amount / actual_risk_per_lot
                else:
                    new_lot_size = ProfessionalComplexConfig.MIN_LOT

                # 如果变化很小，停止迭代
                if abs(new_lot_size - lot_size) < 0.01:
                    lot_size = new_lot_size
                    break
                lot_size = new_lot_size

            # 应用Kelly分数

            kelly_fraction = ProfessionalComplexConfig.RISK_MANAGEMENT['POSITION_SIZING']['KELLY_FRACTION']

            lot_size *= kelly_fraction

            # 波动率调整

            if ProfessionalComplexConfig.RISK_MANAGEMENT['POSITION_SIZING']['VOLATILITY_ADJUSTMENT']:

                indicators = self.data_engine.calculate_complex_indicators()

                atr_percent = indicators.get('ATR_PERCENT', 0.001)

                # 高波动率时减小仓位

                if atr_percent > 0.001:

                    lot_size *= 0.7

                elif atr_percent < 0.0003:

                    lot_size *= 1.1

            correlation_factor = ProfessionalComplexConfig.RISK_MANAGEMENT['POSITION_SIZING']['CORRELATION_FACTOR']
            new_direction = signal.get('direction', 'BUY')

            # 获取当前持仓（通过position_manager，因为需要访问持仓信息）
            # 注意：这里需要从外部传入position_manager或者通过其他方式获取持仓
            # 为了不破坏现有架构，我们通过data_engine访问position_manager
            try:
                # 尝试获取当前持仓信息
                # 由于ComplexRiskManager没有直接访问position_manager，我们需要通过其他方式
                # 最简单的方法是通过MT5直接获取持仓
                current_positions = mt5.positions_get(symbol=self.data_engine.symbol)

                if current_positions:
                    same_direction_count = 0
                    total_positions = len(current_positions)

                    # 统计相同方向的持仓数量
                    for pos in current_positions:
                        existing_direction = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'
                        if existing_direction == new_direction:
                            same_direction_count += 1

                    # 如果存在相同方向的持仓，应用相关性因子
                    if same_direction_count > 0:
                        # 相关性风险：相同方向持仓越多，风险叠加越大
                        # 使用相关性因子降低新仓位，避免风险过度集中

                        # 基础相关性因子（0.8）
                        correlation_multiplier = correlation_factor

                        # 根据相同方向持仓数量动态调整
                        # 1个相同方向持仓：应用基础因子0.8
                        # 2个相同方向持仓：额外减小到0.64 (0.8 * 0.8)
                        # 3个或更多：进一步减小到0.5 (0.8 * 0.8 * 0.78)
                        if same_direction_count == 1:
                            # 1个相同方向持仓，使用基础相关性因子
                            correlation_multiplier = correlation_factor
                        elif same_direction_count == 2:
                            # 2个相同方向持仓，风险叠加更严重，进一步减小
                            correlation_multiplier = correlation_factor * 0.8  # 0.8 * 0.8 = 0.64
                        else:
                            # 3个或更多相同方向持仓，风险高度集中，大幅减小
                            correlation_multiplier = correlation_factor * 0.8 * 0.78  # 约0.5

                        # 确保相关性因子不会太小（至少0.3）
                        correlation_multiplier = max(0.3, correlation_multiplier)

                        lot_size *= correlation_multiplier

                        logger.info(f"📊 相关性调整: 当前有{same_direction_count}个相同方向({new_direction})持仓, "
                                    f"总持仓{total_positions}个, 应用相关性因子{correlation_multiplier:.2f}, "
                                    f"调整后仓位: {lot_size:.2f}手")
                    else:
                        logger.debug(f"📊 相关性检查: 无相同方向持仓，无需调整")

            except Exception as e:
                # 如果获取持仓失败，记录警告但不阻止开仓
                logger.warning(f"⚠️ 获取持仓信息失败，跳过相关性调整: {str(e)}")

            # 根据盈亏比调整仓位：盈亏比越低，手数减少越多
            if ProfessionalComplexConfig.RR_POSITION_ADJUSTMENT and stop_loss and take_profit:
                try:
                    direction = signal.get('direction', 'BUY')
                    # 计算净盈亏比（考虑交易成本）
                    risk_reward_ratio = self.calculate_risk_reward_ratio(
                        entry_price, stop_loss, take_profit, direction, lot_size, include_costs=True
                    )

                    min_rr_for_full = ProfessionalComplexConfig.MIN_RR_FOR_FULL_SIZE
                    min_rr = ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO

                    if risk_reward_ratio < min_rr_for_full:
                        # 盈亏比低于满仓要求，根据盈亏比线性调整
                        # 盈亏比在 min_rr 到 min_rr_for_full 之间时，手数从 0.5 倍到 1.0 倍
                        if risk_reward_ratio >= min_rr:
                            # 线性插值：min_rr -> 0.5倍, min_rr_for_full -> 1.0倍
                            rr_range = min_rr_for_full - min_rr
                            if rr_range > 0:
                                position_multiplier = 0.5 + (risk_reward_ratio - min_rr) / rr_range * 0.5
                            else:
                                position_multiplier = 0.5
                        else:
                            # 盈亏比低于最小要求，大幅减少手数（但不会完全拒绝，因为已经在验证阶段检查过）
                            position_multiplier = 0.3

                        lot_size *= position_multiplier
                        logger.info(
                            f"📊 盈亏比调整: 净盈亏比={risk_reward_ratio:.2f}:1, 仓位倍数={position_multiplier:.2f}, 调整后手数={lot_size:.2f}")
                    else:
                        logger.debug(f"📊 盈亏比充足: {risk_reward_ratio:.2f}:1 >= {min_rr_for_full:.2f}:1, 使用满仓")
                except Exception as e:
                    logger.warning(f"根据盈亏比调整仓位异常: {str(e)}")

            # 限制在合理范围

            lot_size = max(ProfessionalComplexConfig.MIN_LOT,

                           min(ProfessionalComplexConfig.MAX_LOT, lot_size))

            # 四舍五入到步长

            lot_size = round(lot_size / ProfessionalComplexConfig.LOT_STEP) * ProfessionalComplexConfig.LOT_STEP

            return lot_size

        except Exception as e:

            logger.error(f"计算仓位大小异常: {str(e)}")

            return ProfessionalComplexConfig.MIN_LOT

    def _get_support_resistance_levels(self, direction: str, lookback_period: int = 50) -> Tuple[float, float]:
        """获取支撑和阻力位"""
        try:
            prices = list(self.data_engine.price_buffer)
            if len(prices) < lookback_period:
                lookback_period = len(prices)

            if lookback_period > 0:
                recent_prices = prices[-lookback_period:]
                support_level = min(recent_prices)
                resistance_level = max(recent_prices)
                return support_level, resistance_level
            return 0.0, 0.0
        except Exception as e:
            logger.debug(f"获取支撑阻力位异常: {str(e)}")
            return 0.0, 0.0

    def calculate_stop_loss_distance(self, signal: Dict, entry_price: float) -> float:

        """计算止损距离 - 优化版：根据信号强度、市场状态、支撑阻力位动态调整"""

        try:

            indicators = self.data_engine.calculate_complex_indicators()

            atr = indicators.get('ATR', entry_price * 0.001)
            signal_strength = signal.get('strength', 0.5)
            market_state = signal.get('market_state', 'UNCERTAIN')
            adx = indicators.get('ADX', 0)
            atr_percent = indicators.get('ATR_PERCENT', 0)
            direction = signal.get('direction', 'BUY')

            # 强信号（>0.7）：使用更紧的止损（1.0倍ATR），因为信号更可靠，预期价格不会大幅回撤
            # 中等信号（0.5-0.7）：使用标准止损（1.5倍ATR）
            # 弱信号（<0.5）：使用更宽的止损（2.0倍ATR），给市场更多空间，避免被噪音止损
            if signal_strength > 0.7:
                base_atr_multiplier = 1.0
            elif signal_strength > 0.5:
                base_atr_multiplier = 1.5
            else:
                base_atr_multiplier = 2.0

            # 高波动市场需要更宽的止损，避免被正常波动止损
            # 震荡市可以更紧，因为价格波动范围有限
            # 趋势市根据趋势强度调整
            if market_state == 'VOLATILE':
                state_multiplier = 1.3  # 高波动，需要更宽止损
            elif market_state == 'RANGING':
                state_multiplier = 0.9  # 震荡市，可以更紧
            elif market_state == 'TRENDING':
                # 强趋势可以更紧，弱趋势需要更宽
                if adx > 40:
                    state_multiplier = 0.95  # 强趋势，稍微紧一点
                else:
                    state_multiplier = 1.1  # 弱趋势，稍微宽一点
            else:
                state_multiplier = 1.0  # 不确定状态，使用标准值

            # ADX > 50：强趋势，可以更紧
            # ADX < 20：弱趋势或震荡，需要更宽
            if adx > 50:
                adx_multiplier = 0.9
            elif adx < 20:
                adx_multiplier = 1.2
            else:
                adx_multiplier = 1.0

            # 如果ATR百分比很高，说明波动率大，需要更宽止损
            if atr_percent > 0.0015:  # 高波动
                volatility_multiplier = 1.15
            elif atr_percent < 0.0005:  # 低波动
                volatility_multiplier = 0.95
            else:
                volatility_multiplier = 1.0

            atr_multiplier = base_atr_multiplier * state_multiplier * adx_multiplier * volatility_multiplier

            # 限制在合理范围（0.8倍到2.5倍ATR）
            atr_multiplier = max(0.8, min(2.5, atr_multiplier))

            support_level, resistance_level = self._get_support_resistance_levels(direction, 50)
            atr_based_sl_distance = atr * atr_multiplier

            if direction == 'BUY' and support_level > 0:
                # BUY订单：止损应该在支撑位下方
                # 计算到支撑位的距离
                support_distance = entry_price - support_level
                # 使用支撑位下方0.1%或ATR止损，取更合理的
                support_sl_distance = support_distance * 1.1  # 支撑位下方10%的安全边际
                # 取ATR止损和支撑位止损中更紧的（更保守）
                stop_loss_distance = min(atr_based_sl_distance, support_sl_distance)
                logger.debug(
                    f"📊 BUY止损计算: ATR止损={atr_based_sl_distance:.2f}, 支撑位止损={support_sl_distance:.2f}, 最终={stop_loss_distance:.2f}")
            elif direction == 'SELL' and resistance_level > 0:
                # SELL订单：止损应该在阻力位上方
                resistance_distance = resistance_level - entry_price
                # 使用阻力位上方0.1%或ATR止损，取更合理的
                resistance_sl_distance = resistance_distance * 1.1  # 阻力位上方10%的安全边际
                # 取ATR止损和阻力位止损中更紧的（更保守）
                stop_loss_distance = min(atr_based_sl_distance, resistance_sl_distance)
                logger.debug(
                    f"📊 SELL止损计算: ATR止损={atr_based_sl_distance:.2f}, 阻力位止损={resistance_sl_distance:.2f}, 最终={stop_loss_distance:.2f}")
            else:
                # 没有有效的支撑阻力位，使用ATR止损
                stop_loss_distance = atr_based_sl_distance

            # 转换为点数
            point = self.data_engine.data_validator.symbol_info.point if self.data_engine.data_validator.symbol_info else 0.01
            stop_loss_points = stop_loss_distance / point

            # 确保止损距离至少满足最小要求（确保止盈有足够空间）
            # 对于黄金，最小止损距离应该至少2-3美元，这样即使盈亏比1.5:1，止盈也有3-4.5美元
            min_sl_distance_usd = 2.5  # 最小止损距离2.5美元
            min_sl_points = min_sl_distance_usd / point if point > 0 else 0

            if stop_loss_points < min_sl_points:
                stop_loss_points = min_sl_points
                stop_loss_distance = stop_loss_points * point
                logger.info(f"📊 止损距离过小，调整为最小要求: {stop_loss_distance:.2f} ({stop_loss_points:.1f}点)")

            logger.debug(f"📊 止损计算: 信号强度={signal_strength:.2f}, 市场状态={market_state}, ADX={adx:.1f}, "
                         f"ATR倍数={atr_multiplier:.2f}, 止损距离={stop_loss_distance:.2f} ({stop_loss_points:.1f}点)")

            return stop_loss_points

        except Exception as e:

            logger.error(f"计算止损距离异常: {str(e)}")

            return 50  # 默认50点

    def calculate_take_profit_levels(self, signal: Dict, entry_price: float, stop_loss: float) -> List[Dict]:

        """计算止盈目标 - 基于ATR、支撑阻力位、市场状态独立计算，不依赖盈亏比"""

        try:

            signal_strength = signal.get('strength', 0.5)
            market_state = signal.get('market_state', 'UNCERTAIN')
            indicators = self.data_engine.calculate_complex_indicators()
            adx = indicators.get('ADX', 0)
            direction = signal.get('direction', 'BUY')
            atr = indicators.get('ATR', entry_price * 0.001)

            # 判断是否使用动态止盈（单边明确趋势）
            use_dynamic_tp = False
            if (ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['ENABLE'] and
                    ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['USE_FOR_STRONG_TREND_ONLY'] and
                    market_state == 'TRENDING' and
                    adx >= ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['MIN_ADX_FOR_DYNAMIC']):
                use_dynamic_tp = True
                logger.info(f"📈 使用动态止盈（单边明确趋势，ADX={adx:.1f}）")

            # 判断是否使用多目标止盈
            use_multi_target = False
            if ProfessionalComplexConfig.MULTI_TARGET_TP['ENABLE']:
                if (market_state == 'RANGING' and ProfessionalComplexConfig.MULTI_TARGET_TP['USE_FOR_RANGING']):
                    use_multi_target = True
                elif (market_state == 'TRENDING' and adx < ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT[
                    'MIN_ADX_FOR_DYNAMIC'] and
                      ProfessionalComplexConfig.MULTI_TARGET_TP['USE_FOR_WEAK_TREND']):
                    use_multi_target = True
                if use_multi_target:
                    logger.info(f"🎯 使用多目标止盈（市场状态={market_state}，ADX={adx:.1f}）")

            # 获取支撑阻力位
            support_level, resistance_level = self._get_support_resistance_levels(direction, 50)

            # 根据信号强度、市场状态、ADX确定ATR倍数（用于计算止盈距离）
            # 强信号（>0.7）：更高的ATR倍数（3.0-4.0），因为预期盈利空间更大
            # 中等信号（0.5-0.7）：标准ATR倍数（2.0-3.0）
            # 弱信号（<0.5）：较低的ATR倍数（1.5-2.0），但仍保持合理水平
            if signal_strength > 0.7:
                base_atr_multiplier = 3.5
            elif signal_strength > 0.5:
                base_atr_multiplier = 2.5
            else:
                base_atr_multiplier = 2.0

            if market_state == 'TRENDING' and adx > 30:
                # 强趋势市：可以设置更高的止盈，让利润奔跑
                state_multiplier = 1.3
            elif market_state == 'RANGING':
                # 震荡市：保守止盈
                state_multiplier = 0.9
            elif market_state == 'VOLATILE':
                # 高波动市：可以设置更高的止盈
                state_multiplier = 1.2
            else:
                state_multiplier = 1.0

            # ADX > 50：强趋势，可以设置更高的止盈
            # ADX < 20：弱趋势，保守止盈
            if adx > 50:
                adx_multiplier = 1.2
            elif adx < 20:
                adx_multiplier = 0.95
            else:
                adx_multiplier = 1.05

            # 信号强度越高，可以设置更高的止盈
            strength_multiplier = 0.9 + (signal_strength * 0.4)  # 0.9-1.3之间

            atr_multiplier = base_atr_multiplier * state_multiplier * adx_multiplier * strength_multiplier

            # 限制在合理范围（1.5倍到5.0倍ATR）
            atr_multiplier = max(1.5, min(5.0, atr_multiplier))

            # 计算基础止盈距离（基于ATR）
            base_tp_distance = atr * atr_multiplier

            logger.debug(f"📊 止盈ATR倍数计算: 基础={base_atr_multiplier:.2f}, 市场状态倍数={state_multiplier:.2f}, "
                         f"ADX倍数={adx_multiplier:.2f}, 强度倍数={strength_multiplier:.2f}, "
                         f"最终ATR倍数={atr_multiplier:.2f}, 止盈距离=${base_tp_distance:.2f}")

            # 如果使用动态止盈，返回初始止盈目标（后续会动态更新）
            if use_dynamic_tp:
                # 动态止盈：初始止盈使用较小的ATR倍数（2.0倍），后续根据趋势强度动态调整
                initial_atr_mult = 2.0
                initial_tp_distance = atr * initial_atr_mult

                if direction == 'BUY':
                    initial_tp = entry_price + initial_tp_distance
                    # 如果阻力位有效，考虑阻力位
                    if resistance_level > 0 and resistance_level > entry_price:
                        # 如果计算的止盈价格接近阻力位，调整到阻力位附近
                        if abs(initial_tp - resistance_level) < initial_tp_distance * 0.3:
                            initial_tp = resistance_level * 0.998
                        elif initial_tp > resistance_level * 1.01:
                            initial_tp = resistance_level * 1.005
                else:
                    initial_tp = entry_price - initial_tp_distance
                    # 如果支撑位有效，考虑支撑位
                    if support_level > 0 and support_level < entry_price:
                        if abs(initial_tp - support_level) < initial_tp_distance * 0.3:
                            initial_tp = support_level * 1.002
                        elif initial_tp < support_level * 0.99:
                            initial_tp = support_level * 0.995

                logger.info(
                    f"📈 动态止盈：初始止盈={initial_tp:.2f} (ATR倍数={initial_atr_mult:.2f}, 距离=${initial_tp_distance:.2f})，后续将根据趋势强度动态调整")
                return [{'price': initial_tp, 'close_percent': 1.0, 'dynamic': True}]

            # 如果使用多目标止盈，返回两个目标
            if use_multi_target:
                target_count = ProfessionalComplexConfig.MULTI_TARGET_TP['TARGET_COUNT']
                first_target_pct = ProfessionalComplexConfig.MULTI_TARGET_TP['FIRST_TARGET_PCT']
                second_target_pct = ProfessionalComplexConfig.MULTI_TARGET_TP['SECOND_TARGET_PCT']
                first_target_rr = ProfessionalComplexConfig.MULTI_TARGET_TP['FIRST_TARGET_RR']  # 0.8盈亏比
                second_target_rr = ProfessionalComplexConfig.MULTI_TARGET_TP['SECOND_TARGET_RR']  # 1.0盈亏比

                # 获取信号质量评分（用于动态调整止盈目标）
                quality_score = signal.get('quality_score', 0.6)
                success_probability = signal.get('success_probability', 0.5)
                is_trend_start = signal.get('trend_start', False)
                is_reversal = signal.get('reversal_signal', False)

                # 基于信号质量动态调整盈亏比
                # 高质量信号（质量>0.75，成功率>0.7）：提高止盈目标
                # 中等质量信号：使用默认盈亏比
                # 低质量信号：降低止盈目标，更早止盈
                if quality_score >= 0.75 and success_probability >= 0.7:
                    # 高质量信号：提高止盈目标，让利润奔跑
                    quality_multiplier = 1.2
                    tp1_atr_mult = 2.0  # 提高TP1
                    tp2_atr_mult = 2.8  # 提高TP2
                    if is_trend_start:
                        quality_multiplier = 1.3  # 趋势启动信号额外加成
                        tp1_atr_mult = 2.2
                        tp2_atr_mult = 3.0
                elif quality_score >= 0.65 and success_probability >= 0.6:
                    # 中等质量信号：标准设置
                    quality_multiplier = 1.0
                    tp1_atr_mult = 1.5
                    tp2_atr_mult = 2.0
                else:
                    # 低质量信号：降低止盈目标，更早止盈保护
                    quality_multiplier = 0.9
                    tp1_atr_mult = 1.2
                    tp2_atr_mult = 1.6

                # 计算两个止盈目标：基于止损距离和盈亏比
                risk_distance = abs(entry_price - stop_loss)
                tp1_profit = risk_distance * first_target_rr * quality_multiplier  # TP1：动态盈亏比
                tp2_profit = risk_distance * second_target_rr * quality_multiplier  # TP2：动态盈亏比

                # 但也要考虑ATR和支撑阻力位，取更合理的值
                tp1_atr_based = atr * tp1_atr_mult  # TP1基于ATR
                tp2_atr_based = atr * tp2_atr_mult  # TP2基于ATR

                # 取两者中的较大值，确保止盈有足够空间
                tp1_distance = max(tp1_profit, tp1_atr_based)
                tp2_distance = max(tp2_profit, tp2_atr_based)

                # 记录质量调整信息
                logger.info(f"🎯 多目标止盈（质量调整）: 质量评分={quality_score:.2f}, "
                            f"成功率={success_probability:.2%}, 调整倍数={quality_multiplier:.2f}")

                # 计算止盈价格
                if direction == 'BUY':
                    tp1 = entry_price + tp1_distance
                    tp2 = entry_price + tp2_distance

                    # 如果阻力位有效，考虑阻力位
                    if resistance_level > 0 and resistance_level > entry_price:
                        # TP1调整
                        if abs(tp1 - resistance_level) < tp1_distance * 0.3:
                            tp1 = resistance_level * 0.998
                        elif tp1 > resistance_level * 1.01:
                            tp1 = resistance_level * 1.005
                        # TP2调整
                        if abs(tp2 - resistance_level) < tp2_distance * 0.3:
                            tp2 = resistance_level * 0.998
                        elif tp2 > resistance_level * 1.01:
                            tp2 = resistance_level * 1.005
                else:  # SELL
                    tp1 = entry_price - tp1_distance
                    tp2 = entry_price - tp2_distance

                    # 如果支撑位有效，考虑支撑位
                    if support_level > 0 and support_level < entry_price:
                        # TP1调整
                        if abs(tp1 - support_level) < tp1_distance * 0.3:
                            tp1 = support_level * 1.002
                        elif tp1 < support_level * 0.99:
                            tp1 = support_level * 0.995
                        # TP2调整
                        if abs(tp2 - support_level) < tp2_distance * 0.3:
                            tp2 = support_level * 1.002
                        elif tp2 < support_level * 0.99:
                            tp2 = support_level * 0.995

                logger.info(f"🎯 多目标止盈：TP1={tp1:.2f} (距离=${tp1_distance:.2f}, 平仓{first_target_pct:.0%}), "
                            f"TP2={tp2:.2f} (距离=${tp2_distance:.2f}, 平仓{second_target_pct:.0%})")
                return [
                    {'price': tp1, 'close_percent': first_target_pct},
                    {'price': tp2, 'close_percent': second_target_pct}
                ]
            else:
                # 计算单一止盈目标（基于ATR和支撑阻力位）
                if direction == 'BUY':
                    tp_price = entry_price + base_tp_distance

                    # 如果阻力位有效，考虑阻力位
                    if resistance_level > 0 and resistance_level > entry_price:
                        # 如果计算的止盈价格接近阻力位，调整到阻力位附近
                        if abs(tp_price - resistance_level) < base_tp_distance * 0.3:
                            tp_price = resistance_level * 0.998  # 阻力位下方0.2%
                        # 如果计算的止盈价格超过阻力位太多，限制在阻力位附近
                        elif tp_price > resistance_level * 1.01:
                            tp_price = resistance_level * 1.005  # 阻力位上方0.5%
                else:  # SELL
                    tp_price = entry_price - base_tp_distance

                # 如果支撑位有效，考虑支撑位
                if support_level > 0 and support_level < entry_price:
                    # 如果计算的止盈价格接近支撑位，调整到支撑位附近
                    if abs(tp_price - support_level) < base_tp_distance * 0.3:
                        tp_price = support_level * 1.002  # 支撑位上方0.2%
                    # 如果计算的止盈价格低于支撑位太多，限制在支撑位附近
                    elif tp_price < support_level * 0.99:
                        tp_price = support_level * 0.995  # 支撑位下方0.5%

            logger.debug(f"📊 止盈计算: 信号强度={signal_strength:.2f}, 市场状态={market_state}, ADX={adx:.1f}, "
                         f"ATR倍数={atr_multiplier:.2f}, 止盈距离=${base_tp_distance:.2f}, 止盈价格={tp_price:.2f}")

            # 返回单一止盈目标
            return [{'price': tp_price, 'close_percent': 1.0}]

        except Exception as e:

            logger.error(f"计算止盈目标异常: {str(e)}")

            # 异常情况下，尝试返回一个满足最小盈亏比的基本目标
            try:
                min_required_rr = ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO
                risk_distance = abs(entry_price - stop_loss)
                min_profit = risk_distance * min_required_rr
                direction = signal.get('direction', 'BUY')

                if direction == 'BUY':
                    tp1 = entry_price + min_profit
                else:  # SELL
                    tp1 = entry_price - min_profit

                logger.warning(
                    f"⚠️ 止盈计算异常，返回满足最小盈亏比的基本目标: {tp1:.2f} (盈亏比: {min_required_rr:.2f}:1)")
                return [{'price': tp1, 'close_percent': 1.0}]
            except:
                return []

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float,
                                    direction: str, lot_size: float = 1.0,
                                    include_costs: bool = True) -> float:
        """
        计算盈亏比（考虑点差和手续费）

        重要说明：
        - entry_price是开仓价格（BUY用ask，SELL用bid）
        - stop_loss和take_profit是设置的价格（基于entry_price计算）
        - 但实际平仓时，BUY用bid平仓，SELL用ask平仓
        - 因此点差的影响需要额外考虑

        Args:
            entry_price: 入场价格（BUY为ask，SELL为bid）
            stop_loss: 止损设置价格
            take_profit: 止盈设置价格
            direction: 交易方向 ('BUY' 或 'SELL')
            lot_size: 交易手数（用于计算手续费）
            include_costs: 是否考虑交易成本（点差和手续费）

        Returns:
            净盈亏比（考虑交易成本后的实际盈亏比）
        """
        try:
            # 获取点差和每点价值
            symbol_info = self.data_engine.data_validator.symbol_info
            spread = 0.0
            tick_value = ProfessionalComplexConfig.POINT_VALUE
            point = ProfessionalComplexConfig.POINT

            if symbol_info:
                if include_costs and ProfessionalComplexConfig.SPREAD_COST_ENABLED:
                    raw_spread = abs(symbol_info.ask - symbol_info.bid)
                    # 应用点差成本倍数，避免点差过大导致净盈亏比过小
                    spread = raw_spread * ProfessionalComplexConfig.SPREAD_COST_MULTIPLIER
                tick_value = symbol_info.trade_tick_value if symbol_info.trade_tick_value > 0 else ProfessionalComplexConfig.POINT_VALUE
                point = symbol_info.point

            # 计算手续费（只收取一次），转换为价格单位
            commission_in_price = 0.0
            if include_costs:
                total_commission = ProfessionalComplexConfig.COMMISSION_PER_LOT * lot_size  # 只收取一次手续费
                # 将手续费转换为价格单位（美元手续费 / 每点价值 = 点数，再转换为价格）
                commission_in_price = (total_commission / tick_value) * point if tick_value > 0 else 0

            if direction == 'BUY':
                # BUY订单：开仓用ask，平仓用bid
                # 优化：点差只在开仓时发生一次，不应该在风险端和收益端都计入
                # 更合理的方式：将点差成本分摊到风险端和收益端，而不是两端都全额计入
                risk_distance = abs(entry_price - stop_loss)
                reward_distance = abs(take_profit - entry_price)

                if include_costs:
                    # 点差成本：开仓时用ask，平仓时用bid，所以点差影响应该计入
                    # 但为了更合理，将点差成本分摊：50%计入风险，50%计入收益
                    spread_risk = spread * 0.5
                    spread_reward = spread * 0.5
                    risk_distance += spread_risk + commission_in_price
                    reward_distance = max(0, reward_distance - spread_reward - commission_in_price)
            else:  # SELL
                # SELL订单：开仓用bid，平仓用ask
                # 优化：点差只在开仓时发生一次，不应该在风险端和收益端都计入
                risk_distance = abs(stop_loss - entry_price)
                reward_distance = abs(entry_price - take_profit)

                if include_costs:
                    # 点差成本：开仓时用bid，平仓时用ask，所以点差影响应该计入
                    # 但为了更合理，将点差成本分摊：50%计入风险，50%计入收益
                    spread_risk = spread * 0.5
                    spread_reward = spread * 0.5
                    risk_distance += spread_risk + commission_in_price
                    reward_distance = max(0, reward_distance - spread_reward - commission_in_price)

            if risk_distance <= 0:
                return 0.0

            net_rr = reward_distance / risk_distance

            # 添加调试日志
            if include_costs:
                logger.debug(f"📊 净盈亏比计算: 方向={direction}, 入场={entry_price:.2f}, "
                             f"止损={stop_loss:.2f}, 止盈={take_profit:.2f}, "
                             f"点差={spread:.4f}, 手续费={commission_in_price:.4f}, "
                             f"风险距离={risk_distance:.4f}, 收益距离={reward_distance:.4f}, "
                             f"净盈亏比={net_rr:.2f}:1")

            return net_rr
        except Exception as e:
            logger.error(f"计算盈亏比异常: {str(e)}")
            return 0.0

    def validate_risk_reward_ratio(self, signal: Dict, entry_price: float, stop_loss: float,
                                   take_profit: float, lot_size: float = 1.0,
                                   tp_levels: Optional[List[Dict]] = None) -> tuple[bool, float]:
        """
        验证盈亏比是否满足最小要求（考虑交易成本）
        支持多目标止盈的综合净盈亏比计算（不重复计算，直接用综合净盈亏比验证）

        Args:
            signal: 交易信号
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 止盈价格（如果是多目标，这里是最后一个目标或单一目标）
            lot_size: 交易手数（用于计算手续费）
            tp_levels: 多目标止盈列表（如果提供，将计算综合净盈亏比）

        Returns:
            (是否满足要求, 实际净盈亏比)
        """
        try:
            direction = signal.get('direction', 'BUY')

            # 如果有多目标止盈，计算综合净盈亏比（不重复计算）
            if tp_levels and len(tp_levels) > 1:
                # 多目标止盈：计算加权平均止盈价格
                total_close_pct = sum(tp.get('close_percent', 0) for tp in tp_levels)
                if total_close_pct > 0:
                    # 计算加权平均止盈价格
                    weighted_tp = sum(tp['price'] * tp.get('close_percent', 0) for tp in tp_levels) / total_close_pct
                    # 使用加权平均止盈价格计算综合净盈亏比
                    risk_reward_ratio = self.calculate_risk_reward_ratio(
                        entry_price, stop_loss, weighted_tp, direction, lot_size, include_costs=True
                    )
                    logger.info(f"📊 多目标止盈综合净盈亏比: {risk_reward_ratio:.2f}:1 (加权平均止盈={weighted_tp:.2f})")
                else:
                    # 如果总百分比为0，使用最后一个目标
                    risk_reward_ratio = self.calculate_risk_reward_ratio(
                        entry_price, stop_loss, take_profit, direction, lot_size, include_costs=True
                    )
            else:
                # 单一止盈或动态止盈：直接计算净盈亏比
                risk_reward_ratio = self.calculate_risk_reward_ratio(
                    entry_price, stop_loss, take_profit, direction, lot_size, include_costs=True
                )

            min_ratio = ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO

            is_valid = risk_reward_ratio >= min_ratio

            if not is_valid:
                logger.warning(
                    f"⚠️ 净盈亏比不足: {risk_reward_ratio:.2f} < {min_ratio:.2f} (最小要求: {min_ratio:.2f}:1)")

            return is_valid, risk_reward_ratio
        except Exception as e:
            logger.error(f"验证盈亏比异常: {str(e)}")
            return False, 0.0

    def check_risk_limits(self) -> bool:

        """检查风险限制"""

        if not self.account_info:

            self.update_account_info()

            if not self.account_info:
                return False

        # 检查最大回撤

        equity = self.account_info['equity']

        balance = self.account_info['balance']

        drawdown = (balance - equity) / balance if balance > 0 else 0

        if drawdown > ProfessionalComplexConfig.MAX_DRAWDOWN:
            logger.warning(f"⚠️ 回撤超限: {drawdown:.2%} > {ProfessionalComplexConfig.MAX_DRAWDOWN:.2%}")

            return False

        # 检查保证金水平

        margin_level = self.account_info['margin_level']

        if margin_level > 0 and margin_level < 200:
            logger.warning(f"⚠️ 保证金水平过低: {margin_level:.1f}%")

            return False

        return True


class ProfessionalPositionManager:
    """专业仓位管理器 - 处理开仓、平仓和仓位跟踪"""

    def __init__(self, data_engine: ProfessionalTickDataEngine, risk_manager: ComplexRiskManager):

        self.data_engine = data_engine

        self.risk_manager = risk_manager

        self.open_positions = {}

        self.closed_positions = deque(maxlen=100)

        self.daily_trades = 0

        self.last_trade_date = None

        # 存储每个持仓的多目标止盈信息 {ticket: [tp1, tp2, tp3, ...]}

        self.position_tp_targets = {}

        # 记录最近开仓的时间和价格（用于防止在相近价格连开多单）
        self.last_trade_time = 0
        self.last_trade_price = 0.0
        self.last_trade_direction = None  # 'BUY' 或 'SELL'

        # 记录已经设置过止盈止损的订单ticket，避免重复设置
        self.sl_tp_set_positions = set()

        # 盈利回撤控制：记录每笔订单的峰值盈利
        self.position_peak_profit = {}  # {ticket: {'peak_profit_usd': float, 'peak_profit_pct': float}}

        # 动态止盈：记录使用动态止盈的订单
        self.dynamic_tp_positions = set()  # {ticket}

        # 动态止盈更新间隔控制
        self.last_dynamic_tp_update = {}  # {ticket: timestamp}

        # 存储每个订单的信号特征（用于ML训练）
        self.position_signal_features = {}  # {ticket: np.ndarray}

    @staticmethod
    def normalize_price(price: float, digits: int) -> float:
        """规范化价格到指定精度"""
        if digits <= 0:
            return round(price, 2)
        multiplier = 10 ** digits
        return round(price * multiplier) / multiplier

    def _calculate_net_profit(self, position: Dict) -> Tuple[float, float]:
        """
        计算净盈利（扣除手续费和点差）

        Returns:
            (净盈利USD, 净盈利百分比)
        """
        try:
            current_price = position.get('price_current', 0)
            entry_price = position['price_open']
            position_type = position['type']
            volume = position.get('volume', 0)

            if current_price <= 0 or entry_price <= 0:
                return 0.0, 0.0

            # 计算毛盈利
            if position_type == 'BUY':
                gross_profit_usd = (current_price - entry_price) * volume * 100  # 黄金每点约1美元
                gross_profit_pct = (current_price - entry_price) / entry_price
            else:  # SELL
                gross_profit_usd = (entry_price - current_price) * volume * 100
                gross_profit_pct = (entry_price - current_price) / entry_price

            # 计算交易成本
            symbol_info = self.data_engine.data_validator.symbol_info
            if symbol_info:
                # 点差成本（开仓和平仓各一次）
                spread = abs(
                    symbol_info.ask - symbol_info.bid) if ProfessionalComplexConfig.SPREAD_COST_ENABLED else 0.0
                spread_cost = spread * volume * 100 * ProfessionalComplexConfig.SPREAD_COST_MULTIPLIER * 2  # 开仓和平仓

                # 手续费成本（开仓和平仓各一次）
                commission_cost = ProfessionalComplexConfig.COMMISSION_PER_LOT * volume * 2  # 开仓和平仓

                # 净盈利 = 毛盈利 - 交易成本
                net_profit_usd = gross_profit_usd - spread_cost - commission_cost

                # 计算净盈利百分比（基于入场价）
                net_profit_pct = (net_profit_usd / (entry_price * volume * 100)) if (
                                                                                                entry_price * volume * 100) > 0 else 0.0

                return net_profit_usd, net_profit_pct
            else:
                # 如果无法获取品种信息，返回毛盈利
                return gross_profit_usd, gross_profit_pct

        except Exception as e:
            logger.warning(f"计算净盈利异常: {str(e)}")
            return 0.0, 0.0

    def get_open_positions(self) -> Dict:

        """获取当前持仓"""

        try:

            positions = mt5.positions_get(symbol=self.data_engine.symbol)

            new_positions = {}

            if positions:

                for pos in positions:

                    ticket = pos.ticket

                    new_positions[ticket] = {

                        'ticket': ticket,

                        'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',

                        'volume': pos.volume,

                        'price_open': pos.price_open,

                        'price_current': pos.price_current,

                        'profit': pos.profit,

                        'swap': pos.swap,

                        'time': pos.time,

                        'sl': pos.sl,  # 止损价格

                        'tp': pos.tp  # 止盈价格

                    }

                    # 保留已有的多目标止盈信息（如果持仓仍然存在）

                    if ticket in self.position_tp_targets:
                        new_positions[ticket]['tp_targets'] = self.position_tp_targets[ticket]

            # 清理已平仓的持仓的多目标止盈信息和止盈止损设置记录

            closed_tickets = set(self.open_positions.keys()) - set(new_positions.keys())

            for ticket in closed_tickets:

                if ticket in self.position_tp_targets:
                    del self.position_tp_targets[ticket]

                if ticket in self.sl_tp_set_positions:
                    self.sl_tp_set_positions.discard(ticket)

            self.open_positions = new_positions

            return self.open_positions

        except Exception as e:

            logger.error(f"获取持仓异常: {str(e)}")

            return {}

    def _get_filling_mode(self, symbol_info: Any) -> int:

        """获取品种支持的填充模式"""

        try:

            if not symbol_info:
                logger.debug("symbol_info为空，使用默认RETURN填充模式")

                return mt5.ORDER_FILLING_RETURN  # 默认使用RETURN

            # 检查品种支持的填充模式（使用位运算）

            # filling_mode 是一个位掩码，需要与 ORDER_FILLING_* 常量进行位运算

            filling_mode = symbol_info.filling_mode

            logger.debug(f"品种填充模式位掩码: {filling_mode} (二进制: {bin(filling_mode)})")

            # ORDER_FILLING_FOK = 1, ORDER_FILLING_IOC = 2, ORDER_FILLING_RETURN = 4

            # 按优先级尝试：RETURN > IOC > FOK

            # 注意：某些经纪商可能使用不同的位掩码值

            if filling_mode & mt5.ORDER_FILLING_RETURN:

                logger.debug("使用 ORDER_FILLING_RETURN 填充模式")

                return mt5.ORDER_FILLING_RETURN

            elif filling_mode & mt5.ORDER_FILLING_IOC:

                logger.debug("使用 ORDER_FILLING_IOC 填充模式")

                return mt5.ORDER_FILLING_IOC

            elif filling_mode & mt5.ORDER_FILLING_FOK:

                logger.debug("使用 ORDER_FILLING_FOK 填充模式")

                return mt5.ORDER_FILLING_FOK

            else:

                # 如果位运算都不匹配，尝试直接使用填充模式值

                # 某些经纪商可能直接返回填充模式值而不是位掩码

                if filling_mode == mt5.ORDER_FILLING_RETURN:

                    logger.debug("直接匹配 ORDER_FILLING_RETURN")

                    return mt5.ORDER_FILLING_RETURN

                elif filling_mode == mt5.ORDER_FILLING_IOC:

                    logger.debug("直接匹配 ORDER_FILLING_IOC")

                    return mt5.ORDER_FILLING_IOC

                elif filling_mode == mt5.ORDER_FILLING_FOK:

                    logger.debug("直接匹配 ORDER_FILLING_FOK")

                    return mt5.ORDER_FILLING_FOK

                else:

                    # 如果都不支持，使用RETURN（最通用）

                    logger.warning(f"品种填充模式 {filling_mode} 不匹配标准模式，使用默认RETURN模式")

                    return mt5.ORDER_FILLING_RETURN

        except Exception as e:

            logger.warning(f"获取填充模式异常: {str(e)}，使用默认RETURN模式")

            return mt5.ORDER_FILLING_RETURN

    def _get_alternative_filling_mode(self, symbol_info: Any, current_mode: int) -> Optional[int]:

        """获取替代填充模式"""

        try:

            if not symbol_info:
                return mt5.ORDER_FILLING_RETURN

            filling_mode = symbol_info.filling_mode

            logger.debug(f"尝试替代填充模式，当前模式: {current_mode}, 品种支持: {filling_mode}")

            # 按优先级尝试其他支持的填充模式

            # 优先级：RETURN > IOC > FOK

            # 使用 ORDER_FILLING_* 常量进行位运算检查

            if current_mode != mt5.ORDER_FILLING_RETURN and (filling_mode & mt5.ORDER_FILLING_RETURN):

                logger.debug("尝试使用 ORDER_FILLING_RETURN 作为替代")

                return mt5.ORDER_FILLING_RETURN

            elif current_mode != mt5.ORDER_FILLING_IOC and (filling_mode & mt5.ORDER_FILLING_IOC):

                logger.debug("尝试使用 ORDER_FILLING_IOC 作为替代")

                return mt5.ORDER_FILLING_IOC

            elif current_mode != mt5.ORDER_FILLING_FOK and (filling_mode & mt5.ORDER_FILLING_FOK):

                logger.debug("尝试使用 ORDER_FILLING_FOK 作为替代")

                return mt5.ORDER_FILLING_FOK

            # 如果位运算都不匹配，尝试直接值匹配

            if current_mode != mt5.ORDER_FILLING_RETURN and filling_mode == mt5.ORDER_FILLING_RETURN:

                logger.debug("直接匹配 ORDER_FILLING_RETURN 作为替代")

                return mt5.ORDER_FILLING_RETURN

            elif current_mode != mt5.ORDER_FILLING_IOC and filling_mode == mt5.ORDER_FILLING_IOC:

                logger.debug("直接匹配 ORDER_FILLING_IOC 作为替代")

                return mt5.ORDER_FILLING_IOC

            elif current_mode != mt5.ORDER_FILLING_FOK and filling_mode == mt5.ORDER_FILLING_FOK:

                logger.debug("直接匹配 ORDER_FILLING_FOK 作为替代")

                return mt5.ORDER_FILLING_FOK

            # 如果都不支持，返回None

            logger.warning(f"品种不支持任何替代填充模式，当前模式: {current_mode}, 品种支持: {filling_mode}")

            return None

        except Exception as e:

            logger.warning(f"获取替代填充模式异常: {str(e)}")

            return mt5.ORDER_FILLING_RETURN

    def _check_forming_strong_trend(self, indicators: Dict, signal_direction: str) -> tuple[bool, str]:
        """
        检测是否即将形成强单边趋势

        Args:
            indicators: 技术指标字典
            signal_direction: 信号方向 ('BUY' 或 'SELL')

        Returns:
            (是否即将形成强单边趋势, 趋势方向 'BUY'/'SELL'/'NONE')
        """
        try:
            # 1. EMA对齐度较高（>0.5表示有明确方向性）
            ema_alignment = indicators.get('EMA_ALIGNMENT', 0)

            # 2. ADX上升（从低值快速上升，表示趋势正在形成）
            current_adx = indicators.get('ADX', 0)
            previous_adx = indicators.get('ADX_PREV', 0)
            adx_rising = (current_adx > 15 and
                          previous_adx < 15 and
                          current_adx > previous_adx * 1.2)  # ADX从<15快速上升到>15，且增长20%以上

            # 3. 价格动量加速
            prices = list(self.data_engine.price_buffer)
            momentum_acceleration = False
            if len(prices) >= 10:
                momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
                momentum_10 = (prices[-5] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
                momentum_acceleration = abs(momentum_5) > abs(momentum_10) * 1.3  # 动量加速1.3倍以上

            # 4. MACD确认
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_SIGNAL', 0)
            macd_hist = indicators.get('MACD_HIST', 0)

            # 5. 价格突破关键EMA
            current_price = indicators.get('CURRENT_PRICE', 0)
            ema_15 = indicators.get('EMA_15', 0)
            ema_30 = indicators.get('EMA_30', 0)

            # 判断多头趋势形成条件
            bullish_forming = (
                    ema_alignment > 0.5 and  # EMA对齐度较高
                    (adx_rising or momentum_acceleration) and  # ADX上升或动量加速
                    (macd > macd_signal and macd_hist > 0) and  # MACD确认
                    (current_price > ema_15 > ema_30 if ema_15 > 0 and ema_30 > 0 else False)  # 价格突破EMA
            )

            # 判断空头趋势形成条件
            bearish_forming = (
                    ema_alignment < -0.5 and  # EMA对齐度较高（负值）
                    (adx_rising or momentum_acceleration) and  # ADX上升或动量加速
                    (macd < macd_signal and macd_hist < 0) and  # MACD确认
                    (current_price < ema_15 < ema_30 if ema_15 > 0 and ema_30 > 0 else False)  # 价格跌破EMA
            )

            # 检查信号方向是否与即将形成的趋势一致
            if bullish_forming and signal_direction == 'BUY':
                return True, 'BUY'
            elif bearish_forming and signal_direction == 'SELL':
                return True, 'SELL'
            else:
                return False, 'NONE'

        except Exception as e:
            logger.warning(f"检测即将形成强单边趋势异常: {str(e)}")
            return False, 'NONE'

    def can_open_new_position(self, signal: Optional[Dict] = None) -> bool:

        """检查是否可以开新仓"""

        if not signal:
            logger.debug("⏸️ 无信号，无法开仓")
            return False

        new_direction = signal.get('direction')
        signal_strength = signal.get('strength', 0)

        # 检查每日交易限制

        current_date = datetime.now().date()

        if self.last_trade_date != current_date:
            self.daily_trades = 0

            self.last_trade_date = current_date

        if self.daily_trades >= ProfessionalComplexConfig.MAX_DAILY_TRADES:
            logger.warning(
                f"⚠️ [{new_direction}] 达到每日交易限制: {self.daily_trades}/{ProfessionalComplexConfig.MAX_DAILY_TRADES}")

            return False

        # 检查并发持仓限制

        self.get_open_positions()

        if len(self.open_positions) >= ProfessionalComplexConfig.MAX_CONCURRENT_TRADES:
            logger.warning(
                f"⚠️ [{new_direction}] 达到最大并发持仓: {len(self.open_positions)}/{ProfessionalComplexConfig.MAX_CONCURRENT_TRADES}")

            return False

        # 检查风险限制

        if not self.risk_manager.check_risk_limits():
            logger.info(f"⏸️ [{new_direction}] 风险限制检查未通过，无法开仓")

            return False

        # 检查是否已有相反方向的持仓（不允许同时存在多和空）
        # 获取当前持仓
        current_positions = self.get_open_positions()

        # 检查是否有相反方向的持仓
        opposite_positions = []
        for ticket, pos in current_positions.items():
            existing_direction = pos.get('type')  # 'BUY' 或 'SELL'
            if existing_direction != new_direction:
                opposite_positions.append((ticket, pos))

        # 如果存在相反方向的持仓，不允许开仓（已移除反转信号平仓功能）
        if opposite_positions:
            existing_direction = opposite_positions[0][1].get('type')  # 获取第一个相反方向持仓的方向
            logger.warning(f"⚠️ [{new_direction}] 检测到相反方向持仓（{existing_direction}），不允许开仓")
            return False

        # 获取技术指标来判断当前趋势
        indicators = self.data_engine.calculate_complex_indicators()
        if indicators:
            # 优先检查多时间框架EMA趋势排列
            ema_trend = indicators.get('EMA_TREND', 'UNCERTAIN')
            ema_alignment = indicators.get('EMA_ALIGNMENT', 0)

            # 如果EMA排列明确（BULLISH或BEARISH），只允许顺势交易
            if ema_trend in ['BULLISH', 'BEARISH']:
                # 获取EMA趋势详情用于日志
                ema_trend_details = indicators.get('EMA_TREND_ALIGNMENT', {})
                ema_timeframe = ema_trend_details.get('timeframe', 'UNKNOWN')
                ema_details = ema_trend_details.get('details', {})

                # 只允许顺势交易：EMA多头时只允许BUY，EMA空头时只允许SELL
                if ema_trend == 'BULLISH' and new_direction != 'BUY':
                    if ema_timeframe in ema_details:
                        ma_values = ema_details[ema_timeframe]
                        logger.warning(
                            f"❌ [{new_direction}] EMA趋势明确为多头（{ema_timeframe}），但信号方向为{new_direction}，拒绝开仓")
                        logger.warning(
                            f"   EMA详情: MA5={ma_values.get('ma5', 0):.2f} > MA15={ma_values.get('ma15', 0):.2f} > MA30={ma_values.get('ma30', 0):.2f} > MA60={ma_values.get('ma60', 0):.2f}")
                    else:
                        logger.warning(f"❌ [{new_direction}] EMA趋势明确为多头，但信号方向为{new_direction}，拒绝开仓")
                    return False
                elif ema_trend == 'BEARISH' and new_direction != 'SELL':
                    if ema_timeframe in ema_details:
                        ma_values = ema_details[ema_timeframe]
                        logger.warning(
                            f"❌ [{new_direction}] EMA趋势明确为空头（{ema_timeframe}），但信号方向为{new_direction}，拒绝开仓")
                        logger.warning(
                            f"   EMA详情: MA5={ma_values.get('ma5', 0):.2f} < MA15={ma_values.get('ma15', 0):.2f} < MA30={ma_values.get('ma30', 0):.2f} < MA60={ma_values.get('ma60', 0):.2f}")
                    else:
                        logger.warning(f"❌ [{new_direction}] EMA趋势明确为空头，但信号方向为{new_direction}，拒绝开仓")
                    return False
                else:
                    # EMA趋势与信号方向一致，允许开仓
                    if ema_timeframe in ema_details:
                        ma_values = ema_details[ema_timeframe]
                        logger.info(f"✅ [{new_direction}] EMA趋势为{ema_trend}（{ema_timeframe}），信号方向一致，允许开仓")
                        logger.info(
                            f"   EMA详情: MA5={ma_values.get('ma5', 0):.2f}, MA15={ma_values.get('ma15', 0):.2f}, MA30={ma_values.get('ma30', 0):.2f}, MA60={ma_values.get('ma60', 0):.2f}")
                    else:
                        logger.info(f"✅ [{new_direction}] EMA趋势为{ema_trend}，信号方向一致，允许开仓")
            elif ema_trend == 'UNCERTAIN':
                # EMA趋势不明确，检查是否即将形成强单边趋势
                is_forming_strong_trend, trend_direction = self._check_forming_strong_trend(indicators, new_direction)

                if is_forming_strong_trend:
                    # 即将形成强单边趋势，且信号方向与即将形成的趋势一致，允许开小仓
                    logger.info(
                        f"📈 [{new_direction}] EMA即将形成强单边趋势（方向: {trend_direction}），信号方向一致，允许开小仓")
                    # 标记为即将形成趋势，后续会减小仓位
                    signal['forming_strong_trend'] = True
                else:
                    # 完全震荡市，不允许开仓
                    logger.info(
                        f"⏸️ [{new_direction}] EMA趋势为震荡市（UNCERTAIN），且未检测到即将形成强单边趋势，不允许开仓")
                    return False
            else:
                # EMA排列不明确，统一视为震荡市，不允许开仓
                logger.info(f"⏸️ [{new_direction}] EMA趋势不明确，视为震荡市，不允许开仓")
                return False

        # 检查短时间内价格差是否超过10美元（防止在相近价格连开多单）
        # 注意：使用美元价格差而不是点数，因为点数会随手数不同而变化
        # 注意：只有在当前有持仓的情况下才检查价差限制，如果没有持仓则允许开新仓
        if len(self.open_positions) > 0 and self.last_trade_time > 0:
            current_time = time.time()
            time_diff = current_time - self.last_trade_time
            min_time_interval = 180  # 3分钟 = 180秒
            min_price_diff_usd = 10.0  # 最小价差10美元

            if time_diff < min_time_interval:
                # 在3分钟内，检查价差
                current_price = signal.get('entry_price', 0)
                if current_price > 0 and self.last_trade_price > 0:
                    # 直接计算美元价格差
                    price_diff_usd = abs(current_price - self.last_trade_price)

                    if price_diff_usd < min_price_diff_usd:
                        logger.info(f"⏸️ [{new_direction}] 短时间内价差不足: 距离上次开仓 {time_diff:.1f}秒, "
                                    f"价差 ${price_diff_usd:.2f} < ${min_price_diff_usd:.2f} (要求至少10美元价差), "
                                    f"上次价格: {self.last_trade_price:.2f}, 当前价格: {current_price:.2f}, "
                                    f"上次方向: {self.last_trade_direction}")
                        return False
                    else:
                        logger.debug(
                            f"✅ [{new_direction}] 价差检查通过: ${price_diff_usd:.2f} >= ${min_price_diff_usd:.2f}")
            else:
                # 超过3分钟，仍然检查价差（但时间限制更长，比如30分钟内）
                extended_time_interval = 1800  # 30分钟 = 1800秒
                if time_diff < extended_time_interval:
                    current_price = signal.get('entry_price', 0)
                    if current_price > 0 and self.last_trade_price > 0:
                        # 直接计算美元价格差
                        price_diff_usd = abs(current_price - self.last_trade_price)

                        if price_diff_usd < min_price_diff_usd:
                            logger.warning(
                                f"⚠️ [{new_direction}] 30分钟内价差不足: 距离上次开仓 {time_diff / 60:.1f}分钟, "
                                f"价差 ${price_diff_usd:.2f} < ${min_price_diff_usd:.2f} (要求至少10美元价差), "
                                f"上次价格: {self.last_trade_price:.2f}, 当前价格: {current_price:.2f}")
                            return False
                        else:
                            logger.debug(
                                f"✅ [{new_direction}] 价差检查通过: ${price_diff_usd:.2f} >= ${min_price_diff_usd:.2f}")

        # 所有检查都通过
        logger.debug(f"✅ [{new_direction}] 所有开仓检查通过: 强度: {signal_strength:.2f}")
        return True

    def open_position(self, signal: Dict) -> Optional[int]:

        """开仓 - 使用先下单后设置止盈止损的方式"""

        if not self.can_open_new_position(signal):
            # 记录为什么不能开仓（用于调试）
            logger.info(
                f"⏸️ 信号已生成但无法开仓: {signal.get('direction')} 强度: {signal.get('strength', 0):.2f} 价格: {signal.get('entry_price', 0):.2f} - 检查can_open_new_position返回False")
            return None

        try:

            symbol = self.data_engine.symbol

            symbol_info = self.data_engine.data_validator.symbol_info

            if not symbol_info:
                logger.error("无法获取品种信息")

                return None

            # 验证持仓是否真实存在（从MT5重新获取一次，确保数据准确）
            new_direction = signal.get('direction')
            current_positions = self.get_open_positions()

            try:
                mt5_positions = mt5.positions_get(symbol=symbol)
                actual_position_count = len(mt5_positions) if mt5_positions else 0

                if actual_position_count == 0 and len(current_positions) > 0:
                    # MT5显示没有持仓，但缓存中有持仓，说明缓存过期，清空缓存
                    logger.warning(f"⚠️ 持仓缓存不一致：MT5显示无持仓，但缓存中有{len(current_positions)}个持仓，清空缓存")
                    current_positions = {}
                    self.open_positions = {}
                elif actual_position_count > 0 and len(current_positions) == 0:
                    # MT5显示有持仓，但缓存中没有，重新获取
                    logger.info(f"📊 持仓缓存不一致：MT5显示{actual_position_count}个持仓，但缓存为空，重新获取")
                    current_positions = self.get_open_positions()
            except Exception as e:
                logger.warning(f"⚠️ 验证持仓时异常: {str(e)}")

            # 检查是否有相反方向的持仓（已移除反转信号平仓功能，直接拒绝）
            opposite_positions = []
            for ticket, pos in current_positions.items():
                existing_direction = pos.get('type')  # 'BUY' 或 'SELL'
                if existing_direction != new_direction:
                    opposite_positions.append((ticket, pos))

            if opposite_positions:
                existing_direction = opposite_positions[0][1].get('type')
                logger.warning(f"⚠️ 检测到{len(opposite_positions)}个相反方向持仓（{existing_direction}），不允许开仓")
                return None

            # 获取当前价格

            tick = mt5.symbol_info_tick(symbol)

            if not tick:
                logger.error("无法获取当前价格")

                return None

            # 使用安全方法获取tick值

            ask = DataSourceValidator._get_tick_value(tick, 'ask')

            bid = DataSourceValidator._get_tick_value(tick, 'bid')

            entry_price = ask if signal['direction'] == 'BUY' else bid

            order_type = mt5.ORDER_TYPE_BUY if signal['direction'] == 'BUY' else mt5.ORDER_TYPE_SELL

            # 计算止损止盈（但先不下单时设置）

            stop_loss_distance = self.risk_manager.calculate_stop_loss_distance(signal, entry_price)

            point = symbol_info.point

            if signal['direction'] == 'BUY':

                sl_price = entry_price - stop_loss_distance * point

                tp_levels = self.risk_manager.calculate_take_profit_levels(signal, entry_price, sl_price)

                tp_price = tp_levels[0]['price'] if tp_levels else entry_price + stop_loss_distance * point * 2

                # 检查是否使用动态止盈
                use_dynamic_tp = tp_levels and len(tp_levels) > 0 and tp_levels[0].get('dynamic', False)

            else:

                sl_price = entry_price + stop_loss_distance * point

                tp_levels = self.risk_manager.calculate_take_profit_levels(signal, entry_price, sl_price)

                tp_price = tp_levels[0]['price'] if tp_levels else entry_price - stop_loss_distance * point * 2

                # 检查是否使用动态止盈
                use_dynamic_tp = tp_levels and len(tp_levels) > 0 and tp_levels[0].get('dynamic', False)

            # 规范化价格（使用digits精度）
            digits = symbol_info.digits
            sl_price = self.normalize_price(sl_price, digits)
            tp_price = self.normalize_price(tp_price, digits)

            # 规范化tp_levels中的所有价格
            if tp_levels:
                for tp_level in tp_levels:
                    tp_level['price'] = self.normalize_price(tp_level['price'], digits)

            # 检查止盈价与现价之差必须超过4美元
            # 如果计算出的止盈不符合条件，拒绝开仓（不自动调整）
            current_price = entry_price  # 使用入场价作为现价参考
            min_tp_price_diff_usd = 4.0  # 最小止盈价差4美元

            # 检查第一个止盈目标（对于分段止盈，这是TP1；对于单一或动态止盈，这是唯一目标）
            if tp_levels and len(tp_levels) > 0:
                first_tp_price = tp_levels[0]['price']  # 现在已经是规范化的价格
                tp_price_diff_usd = abs(first_tp_price - current_price)

                if tp_price_diff_usd < min_tp_price_diff_usd:
                    logger.warning(f"❌ [{signal['direction']}] 第一个止盈目标价差不足，拒绝开仓: "
                                   f"止盈价差=${tp_price_diff_usd:.2f} < ${min_tp_price_diff_usd:.2f} (要求至少4美元), "
                                   f"入场价={entry_price:.{digits}f}, 第一个止盈价={first_tp_price:.{digits}f}, "
                                   f"止损距离=${abs(entry_price - sl_price):.2f}, "
                                   f"信号盈利能力不足，无法满足最小止盈要求")
                    return None
                else:
                    logger.info(
                        f"✅ [{signal['direction']}] 第一个止盈目标价差验证通过: ${tp_price_diff_usd:.2f} (最小要求: ${min_tp_price_diff_usd:.2f})")
            else:
                # 如果没有tp_levels，使用tp_price检查
                tp_price_diff_usd = abs(tp_price - current_price)
                if tp_price_diff_usd < min_tp_price_diff_usd:
                    logger.warning(f"❌ [{signal['direction']}] 止盈价与现价之差不足，拒绝开仓: "
                                   f"止盈价差=${tp_price_diff_usd:.2f} < ${min_tp_price_diff_usd:.2f} (要求至少4美元), "
                                   f"入场价={entry_price:.{digits}f}, 止盈价={tp_price:.{digits}f}, "
                                   f"止损距离=${abs(entry_price - sl_price):.2f}, "
                                   f"信号盈利能力不足，无法满足最小止盈要求")
                    return None
                logger.info(
                    f"✅ [{signal['direction']}] 止盈价差验证通过: ${tp_price_diff_usd:.2f} (最小要求: ${min_tp_price_diff_usd:.2f})")

            # 对于多目标止盈，计算加权平均止盈价格用于仓位计算和盈亏比验证
            if tp_levels and len(tp_levels) > 1:
                total_close_pct = sum(tp.get('close_percent', 0) for tp in tp_levels)
                if total_close_pct > 0:
                    weighted_tp = sum(tp['price'] * tp.get('close_percent', 0) for tp in tp_levels) / total_close_pct
                    tp_price_for_calc = weighted_tp  # 使用加权平均止盈价格
                    logger.debug(f"📊 多目标止盈：使用加权平均止盈价格={weighted_tp:.{digits}f} 进行仓位和盈亏比计算")
                else:
                    tp_price_for_calc = tp_price
            else:
                tp_price_for_calc = tp_price

            # 初步计算仓位大小用于盈亏比验证（使用正确的止盈价格）
            preliminary_lot_size = self.risk_manager.calculate_position_size(
                signal, entry_price, sl_price, tp_price_for_calc
            )

            # 验证盈亏比：在开仓前验证是否满足最小要求（传递tp_levels用于多目标止盈的综合净盈亏比计算）
            is_valid_rr, actual_rr = self.risk_manager.validate_risk_reward_ratio(
                signal, entry_price, sl_price, tp_price_for_calc, preliminary_lot_size, tp_levels=tp_levels
            )

            if not is_valid_rr:
                tp_display = f"加权平均={tp_price_for_calc:.{digits}f}" if (
                            tp_levels and len(tp_levels) > 1) else f"{tp_price_for_calc:.{digits}f}"
                logger.warning(f"❌ [{signal['direction']}] 盈亏比不足，拒绝开仓: 实际盈亏比={actual_rr:.2f}:1, "
                               f"最小要求={ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO:.2f}:1, "
                               f"入场价={entry_price:.{digits}f}, 止损={sl_price:.{digits}f}, 止盈={tp_display}")
                return None

            logger.info(
                f"✅ [{signal['direction']}] 盈亏比验证通过: {actual_rr:.2f}:1 (最小要求: {ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO:.2f}:1)")

            # 计算最终仓位大小（考虑盈亏比调整，使用正确的止盈价格）
            lot_size = self.risk_manager.calculate_position_size(
                signal, entry_price, sl_price, tp_price_for_calc
            )
            logger.info(f"📊 最终计算仓位: {lot_size:.2f}手 (已考虑交易成本和盈亏比)")

            # 验证止损止盈价格是否符合品种要求
            # 获取最小止损距离（点数）
            # MT5可能使用trade_stops_level属性，如果没有则使用合理的默认值
            stops_level = 0
            try:
                # 尝试使用trade_stops_level属性（这是MT5的标准属性）
                if hasattr(symbol_info, 'trade_stops_level'):
                    stops_level = symbol_info.trade_stops_level
                elif hasattr(symbol_info, 'stops_level'):
                    stops_level = symbol_info.stops_level
            except:
                pass

            # 如果仍然为0，则使用合理的默认值
            # 对于黄金，通常最小止损距离是10-50点，而不是200点，但是挂单或者修改订单止盈止损的话，点差没有超过200点是无法成功的
            if stops_level <= 0:
                current_spread = (symbol_info.ask - symbol_info.bid) / point  # 当前点差（点数）
                # 使用点差的5倍或至少10点，但不超过50点
                stops_level = max(10, min(50, int(current_spread * 5)))
                logger.info(
                    f"⚠️ 品种未提供trade_stops_level，使用计算值: {stops_level}点（当前点差: {current_spread:.1f}点）")
            else:
                logger.info(f"📏 品种最小止损距离: {stops_level}点")

            # 增加安全边际：增加10%的距离，并考虑滑点（最多10点）
            # 降低安全边际和滑点缓冲，避免止损距离过大导致盈亏比过低
            safety_margin = 1.1  # 10%安全边际（降低以增加开仓机会）
            slippage_buffer = 10  # 滑点缓冲（点数，降低以增加开仓机会）
            effective_stops_level = int(stops_level * safety_margin) + slippage_buffer
            logger.info(
                f"🛡️ 应用安全边际: 基础距离={stops_level}点, 安全距离={effective_stops_level}点 (安全边际={safety_margin:.0%}, 滑点缓冲={slippage_buffer}点)")

            if stops_level > 0:

                # 计算止损和止盈距离入场价格的点数

                if signal['direction'] == 'BUY':

                    sl_distance_points = (entry_price - sl_price) / point

                    tp_distance_points = (tp_price - entry_price) / point

                else:

                    sl_distance_points = (sl_price - entry_price) / point

                    tp_distance_points = (entry_price - tp_price) / point

                # 使用安全距离（effective_stops_level）而不是基础距离
                # 计算原始净盈亏比（考虑交易成本），以便调整后保持比例关系
                # 使用初步手数来计算净盈亏比
                preliminary_lot_size = self.risk_manager.calculate_position_size(signal, entry_price, sl_price,
                                                                                 tp_price)
                original_net_rr = self.risk_manager.calculate_risk_reward_ratio(
                    entry_price, sl_price, tp_price, signal['direction'], preliminary_lot_size, include_costs=True
                )

                # 如果没有有效的净盈亏比，计算毛盈亏比作为备选
                original_gross_rr = 0.0
                if sl_distance_points > 0 and tp_distance_points > 0:
                    original_gross_rr = tp_distance_points / sl_distance_points

                # 使用净盈亏比，如果没有则使用毛盈亏比
                target_rr = original_net_rr if original_net_rr > 0 else original_gross_rr
                min_required_rr = ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO
                # 确保目标盈亏比不低于最小要求
                target_rr = max(target_rr, min_required_rr)

                sl_adjusted = False
                tp_adjusted = False

                # 简化调整逻辑：只在必要时做微调，避免多次调整导致盈亏比下降
                if sl_distance_points < effective_stops_level:
                    # 调整止损价格以满足最小距离要求（微调）
                    if signal['direction'] == 'BUY':
                        sl_price = entry_price - effective_stops_level * point
                    else:
                        sl_price = entry_price + effective_stops_level * point

                    # 规范化价格
                    digits = symbol_info.digits
                    sl_price = self.normalize_price(sl_price, digits)
                    sl_adjusted = True

                    # 如果止损被调整，按比例调整止盈以保持盈亏比（微调）
                    if target_rr > 0:
                        new_sl_distance = effective_stops_level
                        # 按目标盈亏比计算止盈距离（简单比例，不做复杂迭代）
                        new_tp_distance = new_sl_distance * target_rr

                        # 确保止盈距离满足最小距离要求
                        if new_tp_distance < effective_stops_level:
                            new_tp_distance = effective_stops_level

                        if signal['direction'] == 'BUY':
                            tp_price = entry_price + new_tp_distance * point
                        else:
                            tp_price = entry_price - new_tp_distance * point

                        tp_price = self.normalize_price(tp_price, digits)
                        tp_adjusted = True
                        logger.debug(
                            f"微调止损止盈: 止损={effective_stops_level}点, 止盈={new_tp_distance:.1f}点, 目标盈亏比={target_rr:.2f}:1")
                    else:
                        logger.debug(f"微调止损价格以满足最小距离要求: {effective_stops_level}点")

                if not tp_adjusted and tp_distance_points < effective_stops_level:
                    # 如果止盈还没被调整，且距离不足，按比例调整止盈（微调）
                    if target_rr > 0 and sl_distance_points >= effective_stops_level:
                        # 根据目标盈亏比调整止盈
                        new_tp_distance = sl_distance_points * target_rr
                        if new_tp_distance < effective_stops_level:
                            new_tp_distance = effective_stops_level

                        if signal['direction'] == 'BUY':
                            tp_price = entry_price + new_tp_distance * point
                        else:
                            tp_price = entry_price - new_tp_distance * point

                        tp_price = self.normalize_price(tp_price, digits)
                        logger.debug(
                            f"微调止盈: 止损={sl_distance_points:.1f}点, 止盈={new_tp_distance:.1f}点, 目标盈亏比={target_rr:.2f}:1")
                    else:
                        # 如果无法保持盈亏比，至少满足最小距离要求
                        if signal['direction'] == 'BUY':
                            tp_price = entry_price + effective_stops_level * point
                        else:
                            tp_price = entry_price - effective_stops_level * point
                        tp_price = self.normalize_price(tp_price, digits)
                        logger.debug(f"微调止盈价格以满足最小距离要求: {effective_stops_level}点")

            # 验证止损止盈价格方向是否正确

            if signal['direction'] == 'BUY':

                if sl_price >= entry_price:
                    logger.warning(f"⚠️ 止损价格无效（BUY订单止损应低于入场价），跳过设置止损")

                    sl_price = 0

                if tp_price <= entry_price:
                    logger.warning(f"⚠️ 止盈价格无效（BUY订单止盈应高于入场价），跳过设置止盈")

                    tp_price = 0

            else:  # SELL

                if sl_price <= entry_price:
                    logger.warning(f"⚠️ 止损价格无效（SELL订单止损应高于入场价），跳过设置止损")

                    sl_price = 0

                if tp_price >= entry_price:
                    logger.warning(f"⚠️ 止盈价格无效（SELL订单止盈应低于入场价），跳过设置止盈")

                    tp_price = 0

            # 再次验证盈亏比（在调整止损止盈后）
            if sl_price > 0 and tp_price > 0:
                # 如果有多目标止盈且tp_price被调整，需要同步调整tp_levels
                if tp_levels and len(tp_levels) > 1 and tp_adjusted:
                    # 计算调整比例（基于第一个止盈目标）
                    original_first_tp = tp_levels[0]['price']
                    if original_first_tp > 0:
                        # 计算调整比例（按比例调整所有目标）
                        if signal['direction'] == 'BUY':
                            # BUY: 止盈价格增加，所有目标按比例增加
                            adjustment_ratio = (tp_price - entry_price) / (original_first_tp - entry_price) if (
                                                                                                                           original_first_tp - entry_price) > 0 else 1.0
                            for tp_level in tp_levels:
                                original_tp = tp_level['price']
                                new_tp = entry_price + (original_tp - entry_price) * adjustment_ratio
                                tp_level['price'] = self.normalize_price(new_tp, digits)
                        else:  # SELL
                            # SELL: 止盈价格减少，所有目标按比例减少
                            adjustment_ratio = (entry_price - tp_price) / (entry_price - original_first_tp) if (
                                                                                                                           entry_price - original_first_tp) > 0 else 1.0
                            for tp_level in tp_levels:
                                original_tp = tp_level['price']
                                new_tp = entry_price - (entry_price - original_tp) * adjustment_ratio
                                tp_level['price'] = self.normalize_price(new_tp, digits)
                        logger.debug(f"📊 多目标止盈已同步调整: 调整比例={adjustment_ratio:.2f}")

                # 重新计算加权平均止盈价格（如果有多目标止盈）
                if tp_levels and len(tp_levels) > 1:
                    total_close_pct = sum(tp.get('close_percent', 0) for tp in tp_levels)
                    if total_close_pct > 0:
                        weighted_tp = sum(
                            tp['price'] * tp.get('close_percent', 0) for tp in tp_levels) / total_close_pct
                        tp_price_for_calc = weighted_tp
                        logger.debug(f"📊 调整后多目标止盈：使用加权平均止盈价格={weighted_tp:.{digits}f}")
                    else:
                        tp_price_for_calc = tp_price
                else:
                    tp_price_for_calc = tp_price

                # 重新计算手数（因为止损止盈可能已调整）
                lot_size = self.risk_manager.calculate_position_size(signal, entry_price, sl_price, tp_price_for_calc)

                # 使用调整后的手数和止盈价格验证净盈亏比（对于多目标止盈，传递tp_levels）
                is_valid_rr, actual_net_rr = self.risk_manager.validate_risk_reward_ratio(
                    signal, entry_price, sl_price, tp_price_for_calc, lot_size, tp_levels=tp_levels
                )

                min_required_rr = ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO
                if not is_valid_rr:
                    tp_display = f"加权平均={tp_price_for_calc:.{digits}f}" if (
                                tp_levels and len(tp_levels) > 1) else f"{tp_price_for_calc:.{digits}f}"
                    logger.warning(
                        f"❌ [{signal['direction']}] 调整止损止盈后净盈亏比不足，拒绝开仓: 实际净盈亏比={actual_net_rr:.2f}:1, "
                        f"最小要求={min_required_rr:.2f}:1, 止盈={tp_display}")
                    return None

                logger.debug(f"✅ [{signal['direction']}] 调整后净盈亏比验证通过: {actual_net_rr:.2f}:1")
                logger.info(f"📊 调整后重新计算仓位: {lot_size:.2f}手, 净盈亏比={actual_net_rr:.2f}:1")

            # 第一步：先下单，不设置填充模式和止盈止损（避免填充模式问题）

            request = {

                "action": mt5.TRADE_ACTION_DEAL,

                "symbol": symbol,

                "volume": lot_size,

                "type": order_type,

                "price": entry_price,

                # 不设置 sl 和 tp，让MT5使用默认值（无止盈止损）

                "deviation": 20,

                "magic": 123456,

                "comment": f"Auto_{signal['direction']}",

                "type_time": mt5.ORDER_TIME_GTC,

                # 不设置 type_filling，让MT5使用默认填充模式

            }

            result = mt5.order_send(request)

            # 检查返回值是否为None

            if result is None:
                error_code = mt5.last_error()

                logger.error(f"开仓失败: order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")

                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                # 特殊处理：自动交易被禁用
                if result.retcode == 10027:  # TRADE_RETCODE_AUTOTRADING_DISABLED
                    logger.error(f"❌ 开仓失败: MT5自动交易被禁用 (错误代码: {result.retcode})")
                    logger.error(f"💡 解决方案: 请在MT5客户端中启用自动交易")
                    logger.error(f"   1. 打开MT5客户端")
                    logger.error(f"   2. 点击工具栏上的'自动交易'按钮（或按快捷键 Ctrl+E）")
                    logger.error(f"   3. 确保按钮显示为绿色（已启用）")
                    logger.error(f"   4. 或者：工具 -> 选项 -> 专家顾问 -> 勾选'允许自动交易'")
                else:
                    logger.error(f"开仓失败: {result.retcode} - {result.comment}")

                return None

            # 获取实际成交的订单号

            order_ticket = result.order

            if not order_ticket:
                logger.error("开仓成功但未获取到订单号")

                return None

            logger.info(f"✅ 开仓成功: {signal['direction']} {lot_size}手 @ {entry_price:.2f} (订单号: {order_ticket})")

            # 更新最近开仓的时间和价格
            self.last_trade_time = time.time()
            self.last_trade_price = entry_price
            self.last_trade_direction = signal['direction']

            # 保存信号特征用于ML训练（如果信号包含ML评估）
            if 'ml_evaluation' in signal and hasattr(self, 'signal_generator'):
                try:
                    indicators = self.data_engine.calculate_complex_indicators()
                    market_state = signal.get('market_state', 'UNCERTAIN')
                    state_confidence = signal.get('state_confidence', 0.5)
                    signal_features = self.signal_generator.ml_evaluator.extract_features(
                        signal, indicators, market_state, state_confidence, self.data_engine
                    )
                    # 等待持仓建立后，保存到position_signal_features
                    time.sleep(0.2)
                    positions = mt5.positions_get(symbol=symbol)
                    if positions:
                        for pos in positions:
                            if (hasattr(pos, 'identifier') and pos.identifier == order_ticket) or \
                                    (pos.type == order_type and abs(pos.price_open - entry_price) < point * 10):
                                self.position_signal_features[pos.ticket] = signal_features
                                logger.debug(f"📊 保存信号特征用于ML训练: ticket={pos.ticket}")
                                break
                except Exception as e:
                    logger.warning(f"保存信号特征异常: {str(e)}")

            # 检查是否使用动态止盈
            use_dynamic_tp = tp_levels and len(tp_levels) > 0 and tp_levels[0].get('dynamic', False)

            # 保存多目标止盈信息（非动态止盈）

            if tp_levels and len(tp_levels) > 0 and not use_dynamic_tp:

                # 规范化所有止盈目标价格，并验证最小距离
                # 获取最小止损距离（与上面相同的逻辑）
                min_stops_level = 0
                try:
                    if hasattr(symbol_info, 'trade_stops_level'):
                        min_stops_level = symbol_info.trade_stops_level
                    elif hasattr(symbol_info, 'stops_level'):
                        min_stops_level = symbol_info.stops_level
                except:
                    pass

                if min_stops_level <= 0:
                    current_spread = (symbol_info.ask - symbol_info.bid) / point
                    min_stops_level = max(2, int(current_spread * 2))

                normalized_tp_levels = []

                for tp_level in tp_levels:

                    # 规范化价格（使用digits精度）
                    digits = symbol_info.digits
                    normalized_price = self.normalize_price(tp_level['price'], digits)

                    # 验证止盈目标是否满足最小距离要求
                    if signal['direction'] == 'BUY':
                        tp_distance = (normalized_price - entry_price) / point
                    else:
                        tp_distance = (entry_price - normalized_price) / point

                    # 如果距离不足，调整价格
                    if tp_distance < min_stops_level:
                        if signal['direction'] == 'BUY':
                            normalized_price = entry_price + min_stops_level * point
                        else:
                            normalized_price = entry_price - min_stops_level * point
                        normalized_price = self.normalize_price(normalized_price, digits)
                        logger.debug(f"调整止盈目标价格以满足最小距离要求: {min_stops_level}点")

                    normalized_tp_levels.append({

                        'price': normalized_price,

                        'close_percent': tp_level['close_percent']

                    })

                # 等待持仓建立后，保存到position_tp_targets

                time.sleep(0.3)  # 等待持仓完全建立

                positions = mt5.positions_get(symbol=symbol)

                if positions:

                    for pos in positions:
                        if (hasattr(pos, 'identifier') and pos.identifier == order_ticket) or \
                                (pos.type == order_type and abs(pos.price_open - entry_price) < point * 10):
                            self.position_tp_targets[pos.ticket] = normalized_tp_levels
                            logger.info(f"📊 保存多目标止盈: {len(normalized_tp_levels)}个目标")
                            break

            # 如果止损或止盈都无效，跳过设置

            if sl_price == 0 and tp_price == 0:
                logger.warning(f"⚠️ 止损和止盈价格都无效，跳过设置")

                self.daily_trades += 1

                return order_ticket

            # 第二步：立即设置止盈止损（尽可能快）
            # 订单成交后立即尝试设置，不等待，通过多次尝试来确保持仓建立
            position_ticket = None
            actual_position = None
            max_find_attempts = 10  # 增加尝试次数，但每次等待时间更短
            find_attempt_interval = 0.05  # 每次只等待0.05秒，更快响应

            for attempt in range(max_find_attempts):
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    for pos in positions:
                        # 通过订单号或价格匹配找到对应的持仓
                        # 优先使用identifier匹配，如果没有则使用价格和类型匹配
                        if hasattr(pos, 'identifier') and pos.identifier == order_ticket:
                            position_ticket = pos.ticket
                            actual_position = pos
                            logger.info(
                                f"✅ 通过identifier找到持仓: ticket={position_ticket}, order_ticket={order_ticket} (尝试 {attempt + 1})")
                            break
                        elif pos.type == order_type and abs(pos.price_open - entry_price) < point * 10:
                            # 检查是否已经匹配过（避免重复匹配）
                            if position_ticket is None or position_ticket != pos.ticket:
                                position_ticket = pos.ticket
                                actual_position = pos
                                logger.info(
                                    f"✅ 通过价格匹配找到持仓: ticket={position_ticket}, 入场价={pos.price_open:.{symbol_info.digits}f} (尝试 {attempt + 1})")
                            break

                    if position_ticket:
                        break

                if attempt < max_find_attempts - 1:
                    time.sleep(find_attempt_interval)  # 只等待0.05秒

            if not position_ticket:

                # 如果找不到持仓，尝试使用订单号（某些情况下可能相同）

                logger.warning(f"⚠️ 未找到对应持仓，尝试使用订单号: {order_ticket}")
                position_ticket = order_ticket

                # 再次尝试获取持仓信息
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    for pos in positions:
                        if pos.ticket == position_ticket:
                            actual_position = pos
                            break

            # 使用 OrderModify 设置止盈止损
            # 获取实际持仓信息，使用实际入场价格重新验证止盈止损
            if not actual_position:
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    for pos in positions:
                        if pos.ticket == position_ticket:
                            actual_position = pos
                            break

            actual_entry_price = entry_price
            current_sl = 0
            current_tp = 0

            if actual_position:
                actual_entry_price = actual_position.price_open
                current_sl = actual_position.sl if hasattr(actual_position, 'sl') else 0
                current_tp = actual_position.tp if hasattr(actual_position, 'tp') else 0
                logger.info(
                    f"📋 当前持仓信息: ticket={position_ticket}, 入场价={actual_entry_price:.{symbol_info.digits}f}, "
                    f"当前SL={current_sl:.{symbol_info.digits}f}, 当前TP={current_tp:.{symbol_info.digits}f}")

            # 使用实际入场价格重新验证和调整止盈止损
            point = symbol_info.point
            digits = symbol_info.digits
            stops_level = 0
            try:
                if hasattr(symbol_info, 'trade_stops_level'):
                    stops_level = symbol_info.trade_stops_level
                elif hasattr(symbol_info, 'stops_level'):
                    stops_level = symbol_info.stops_level
            except:
                pass

            if stops_level <= 0:
                current_spread = (symbol_info.ask - symbol_info.bid) / point
                stops_level = max(10, min(50, int(current_spread * 5)))

            # 应用安全边际
            safety_margin = 1.2  # 20%安全边际
            slippage_buffer = 20  # 滑点缓冲（点数）
            effective_stops_level = int(stops_level * safety_margin) + slippage_buffer

            logger.info(
                f"🔍 验证止盈止损: 入场价={actual_entry_price:.{digits}f}, 方向={signal['direction']}, 基础距离={stops_level}点, 安全距离={effective_stops_level}点, point={point}, digits={digits}")
            logger.info(f"🔍 初始价格: SL={sl_price:.{digits}f}, TP={tp_price:.{digits}f}")

            if actual_position:
                # 重新验证止损（使用安全距离）
                if sl_price > 0:
                    if signal['direction'] == 'BUY':
                        sl_distance = (actual_entry_price - sl_price) / point
                        logger.info(f"🔍 BUY止损验证: 距离={sl_distance:.1f}点, 要求>={effective_stops_level}点")
                        if sl_price >= actual_entry_price or sl_distance < effective_stops_level:
                            old_sl = sl_price
                            sl_price = actual_entry_price - effective_stops_level * point
                            sl_price = self.normalize_price(sl_price, digits)
                            logger.info(
                                f"✅ 调整止损: {old_sl:.{digits}f} -> {sl_price:.{digits}f} (距离: {effective_stops_level}点)")
                    else:  # SELL
                        sl_distance = (sl_price - actual_entry_price) / point
                        logger.info(f"🔍 SELL止损验证: 距离={sl_distance:.1f}点, 要求>={effective_stops_level}点")
                        if sl_price <= actual_entry_price or sl_distance < effective_stops_level:
                            old_sl = sl_price
                            sl_price = actual_entry_price + effective_stops_level * point
                            sl_price = self.normalize_price(sl_price, digits)
                            logger.info(
                                f"✅ 调整止损: {old_sl:.{digits}f} -> {sl_price:.{digits}f} (距离: {effective_stops_level}点)")

                    # 最终验证止损方向
                    if signal['direction'] == 'BUY' and sl_price >= actual_entry_price:
                        logger.warning(
                            f"⚠️ 止损价格无效（BUY订单止损应低于入场价 {actual_entry_price:.{digits}f}），跳过设置止损")
                        sl_price = 0
                    elif signal['direction'] == 'SELL' and sl_price <= actual_entry_price:
                        logger.warning(
                            f"⚠️ 止损价格无效（SELL订单止损应高于入场价 {actual_entry_price:.{digits}f}），跳过设置止损")
                        sl_price = 0

                # 重新验证止盈（使用安全距离）
                if tp_price > 0:
                    if signal['direction'] == 'BUY':
                        tp_distance = (tp_price - actual_entry_price) / point
                        logger.info(f"🔍 BUY止盈验证: 距离={tp_distance:.1f}点, 要求>={effective_stops_level}点")
                        if tp_price <= actual_entry_price or tp_distance < effective_stops_level:
                            old_tp = tp_price
                            tp_price = actual_entry_price + effective_stops_level * point
                            tp_price = self.normalize_price(tp_price, digits)
                            logger.info(
                                f"✅ 调整止盈: {old_tp:.{digits}f} -> {tp_price:.{digits}f} (距离: {effective_stops_level}点)")
                    else:  # SELL
                        tp_distance = (actual_entry_price - tp_price) / point
                        logger.info(f"🔍 SELL止盈验证: 距离={tp_distance:.1f}点, 要求>={effective_stops_level}点")
                        if tp_price >= actual_entry_price or tp_distance < effective_stops_level:
                            old_tp = tp_price
                            tp_price = actual_entry_price - effective_stops_level * point
                            tp_price = self.normalize_price(tp_price, digits)
                            logger.info(
                                f"✅ 调整止盈: {old_tp:.{digits}f} -> {tp_price:.{digits}f} (距离: {effective_stops_level}点)")

                    # 最终验证止盈方向
                    if signal['direction'] == 'BUY' and tp_price <= actual_entry_price:
                        logger.warning(
                            f"⚠️ 止盈价格无效（BUY订单止盈应高于入场价 {actual_entry_price:.{digits}f}），跳过设置止盈")
                        tp_price = 0
                    elif signal['direction'] == 'SELL' and tp_price >= actual_entry_price:
                        logger.warning(
                            f"⚠️ 止盈价格无效（SELL订单止盈应低于入场价 {actual_entry_price:.{digits}f}），跳过设置止盈")
                        tp_price = 0

                # 最终规范化价格
                if sl_price > 0:
                    sl_price = self.normalize_price(sl_price, digits)
                if tp_price > 0:
                    tp_price = self.normalize_price(tp_price, digits)

                # 最终验证盈亏比（使用实际入场价格）
                if sl_price > 0 and tp_price > 0:
                    # 使用实际入场价格重新计算手数
                    final_lot_size = self.risk_manager.calculate_position_size(
                        signal, actual_entry_price, sl_price, tp_price
                    )
                    is_valid_rr, actual_rr = self.risk_manager.validate_risk_reward_ratio(
                        signal, actual_entry_price, sl_price, tp_price, final_lot_size
                    )

                    if not is_valid_rr:
                        logger.warning(
                            f"❌ [{signal['direction']}] 使用实际入场价格后净盈亏比不足，尝试调整止盈: 实际净盈亏比={actual_rr:.2f}:1, "
                            f"最小要求={ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO:.2f}:1, "
                            f"实际入场价={actual_entry_price:.{digits}f}, 止损={sl_price:.{digits}f}, 止盈={tp_price:.{digits}f}")
                        # 如果净盈亏比不足，尝试调整止盈价格以满足最小净盈亏比要求
                        # 使用迭代方法，因为交易成本依赖于价格距离
                        min_rr = ProfessionalComplexConfig.MIN_RISK_REWARD_RATIO
                        max_iterations = 5
                        iteration = 0
                        current_tp = tp_price
                        found_valid_tp = False

                        while iteration < max_iterations:
                            # 计算当前净盈亏比
                            current_net_rr = self.risk_manager.calculate_risk_reward_ratio(
                                actual_entry_price, sl_price, current_tp, signal['direction'], final_lot_size,
                                include_costs=True
                            )

                            if current_net_rr >= min_rr:
                                found_valid_tp = True
                                break

                            # 计算需要增加的止盈距离
                            # 使用安全系数来补偿交易成本的影响
                            if current_net_rr > 0:
                                safety_factor = min_rr / current_net_rr
                            else:
                                safety_factor = 1.5  # 如果净盈亏比为0，使用保守估计

                            # 计算当前止盈距离
                            if signal['direction'] == 'BUY':
                                current_tp_distance = (current_tp - actual_entry_price) / point
                                new_tp_distance = current_tp_distance * safety_factor
                                new_tp_price = actual_entry_price + new_tp_distance * point
                            else:  # SELL
                                current_tp_distance = (actual_entry_price - current_tp) / point
                                new_tp_distance = current_tp_distance * safety_factor
                                new_tp_price = actual_entry_price - new_tp_distance * point

                            # 确保新止盈价格满足最小距离要求
                            if signal['direction'] == 'BUY':
                                tp_distance = (new_tp_price - actual_entry_price) / point
                            else:
                                tp_distance = (actual_entry_price - new_tp_price) / point

                            if tp_distance >= effective_stops_level:
                                current_tp = self.normalize_price(new_tp_price, digits)
                                iteration += 1
                            else:
                                # 如果无法满足最小距离要求，使用最小距离
                                if signal['direction'] == 'BUY':
                                    current_tp = actual_entry_price + effective_stops_level * point
                                else:
                                    current_tp = actual_entry_price - effective_stops_level * point
                                current_tp = self.normalize_price(current_tp, digits)

                                # 验证使用最小距离后的净盈亏比
                                final_net_rr = self.risk_manager.calculate_risk_reward_ratio(
                                    actual_entry_price, sl_price, current_tp, signal['direction'], final_lot_size,
                                    include_costs=True
                                )

                                if final_net_rr >= min_rr:
                                    found_valid_tp = True
                                break

                        if found_valid_tp:
                            tp_price = current_tp
                            final_net_rr = self.risk_manager.calculate_risk_reward_ratio(
                                actual_entry_price, sl_price, tp_price, signal['direction'], final_lot_size,
                                include_costs=True
                            )
                            logger.info(
                                f"🔧 调整止盈价格以满足最小净盈亏比: {tp_price:.{digits}f} (净盈亏比: {final_net_rr:.2f}:1, 迭代{iteration}次)")
                        else:
                            logger.warning(
                                f"⚠️ 无法调整止盈价格以满足净盈亏比（会违反最小距离要求或超过最大迭代次数），跳过设置止盈")
                            tp_price = 0
                    else:
                        logger.info(f"✅ 最终盈亏比验证通过: {actual_rr:.2f}:1")

                logger.info(f"✅ 最终止盈止损: SL={sl_price:.{digits}f}, TP={tp_price:.{digits}f}")
            else:
                logger.warning(f"⚠️ 无法获取持仓信息，使用原始价格")

            # 只设置有效的止损和止盈
            if sl_price == 0 and tp_price == 0:
                logger.warning(f"⚠️ 止损和止盈都无效，跳过设置")
                self.daily_trades += 1
                return order_ticket

            # 检查是否找到持仓ticket
            if not position_ticket:
                logger.warning(f"⚠️ 未找到持仓ticket，无法设置止盈止损")
                self.daily_trades += 1
                return order_ticket

            # 检查是否已经为该持仓设置过止盈止损，避免重复设置
            if position_ticket in self.sl_tp_set_positions:
                logger.info(f"ℹ️ 持仓 {position_ticket} 已经设置过止盈止损，跳过重复设置")
                self.daily_trades += 1
                return order_ticket

            # 确保价格规范化
            digits = symbol_info.digits
            if sl_price > 0:
                sl_price = self.normalize_price(sl_price, digits)
            if tp_price > 0:
                tp_price = self.normalize_price(tp_price, digits)

            modify_request = {

                "action": mt5.TRADE_ACTION_SLTP,

                "symbol": symbol,

                "position": position_ticket,

            }

            if sl_price > 0:
                modify_request["sl"] = sl_price

            if tp_price > 0:
                modify_request["tp"] = tp_price

            logger.info(
                f"📤 发送止盈止损设置请求: position={position_ticket}, SL={modify_request.get('sl', 0):.{digits}f}, TP={modify_request.get('tp', 0):.{digits}f}")
            modify_result = mt5.order_send(modify_request)

            # 如果设置失败，使用最新价格重新计算并重试
            max_retries = 3  # 增加重试次数
            retry_count = 0
            setup_success = False

            while retry_count < max_retries:
                if modify_result is None:
                    error_code = mt5.last_error()
                    logger.warning(
                        f"⚠️ 止盈止损设置失败 (尝试 {retry_count + 1}/{max_retries}): order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")
                elif modify_result.retcode != mt5.TRADE_RETCODE_DONE:
                    error_code = modify_result.retcode
                    error_comment = modify_result.comment
                    logger.warning(
                        f"⚠️ 止盈止损设置失败 (尝试 {retry_count + 1}/{max_retries}): {error_code} - {error_comment}")

                    # 特殊处理错误代码 10025 "No changes"
                    if error_code == 10025:
                        logger.info(f"🔄 检测到错误10025 (No changes)，检查当前持仓的止盈止损...")
                        # 重新获取持仓信息
                        positions = mt5.positions_get(symbol=symbol)
                        if positions:
                            for pos in positions:
                                if pos.ticket == position_ticket:
                                    current_sl = pos.sl if hasattr(pos, 'sl') else 0
                                    current_tp = pos.tp if hasattr(pos, 'tp') else 0
                                    logger.info(
                                        f"📋 当前持仓止盈止损: SL={current_sl:.{digits}f}, TP={current_tp:.{digits}f}")

                                    # 如果当前止盈止损和我们要设置的值相同，说明已经设置成功了
                                    if abs(current_sl - sl_price) < point * 0.1 and abs(
                                            current_tp - tp_price) < point * 0.1:
                                        logger.info(
                                            f"✅ 止盈止损已存在且值相同，视为设置成功: SL:{sl_price:.{digits}f} TP:{tp_price:.{digits}f}")
                                        setup_success = True
                                        # 记录已设置止盈止损的持仓，避免重复设置
                                        self.sl_tp_set_positions.add(position_ticket)
                                        break
                                    else:
                                        # 如果值不同，调整价格后重试
                                        logger.info(f"🔄 当前止盈止损值与请求不同，调整后重试...")
                                        # 如果当前有止盈止损，我们需要设置不同的值
                                        if current_sl > 0 and abs(current_sl - sl_price) < point * 0.1:
                                            # 当前止损和我们要设置的值太接近，调整
                                            if signal['direction'] == 'BUY':
                                                sl_price = actual_entry_price - (effective_stops_level + 10) * point
                                            else:
                                                sl_price = actual_entry_price + (effective_stops_level + 10) * point
                                            sl_price = self.normalize_price(sl_price, digits)

                                        if current_tp > 0 and abs(current_tp - tp_price) < point * 0.1:
                                            # 当前止盈和我们要设置的值太接近，调整
                                            if signal['direction'] == 'BUY':
                                                tp_price = actual_entry_price + (effective_stops_level + 10) * point
                                            else:
                                                tp_price = actual_entry_price - (effective_stops_level + 10) * point
                                            tp_price = self.normalize_price(tp_price, digits)

                                        logger.info(
                                            f"🔄 调整后的止盈止损: SL={sl_price:.{digits}f}, TP={tp_price:.{digits}f}")
                                    break

                        if setup_success:
                            break
                else:
                    logger.info(f"✅ 止盈止损设置成功: SL:{sl_price:.{digits}f} TP:{tp_price:.{digits}f}")
                    setup_success = True
                    # 记录已设置止盈止损的持仓，避免重复设置
                    self.sl_tp_set_positions.add(position_ticket)
                    break

                # 如果失败，重新获取最新价格并重新计算
                if retry_count < max_retries - 1 and not setup_success:
                    logger.info(f"🔄 重新获取最新价格并重新计算止盈止损...")
                    time.sleep(0.3)  # 等待价格更新

                    # 重新获取最新价格和持仓信息
                    current_tick = mt5.symbol_info_tick(symbol)
                    positions = mt5.positions_get(symbol=symbol)

                    if not current_tick or not positions:
                        logger.warning(f"⚠️ 无法获取最新价格或持仓信息，放弃重试")
                        break

                    # 获取最新价格
                    current_ask = DataSourceValidator._get_tick_value(current_tick, 'ask')
                    current_bid = DataSourceValidator._get_tick_value(current_tick, 'bid')
                    current_spread_points = (current_ask - current_bid) / point

                    # 找到对应的持仓
                    actual_position = None
                    for pos in positions:
                        if pos.ticket == position_ticket:
                            actual_position = pos
                            break

                    if not actual_position:
                        logger.warning(f"⚠️ 未找到持仓 {position_ticket}，放弃重试")
                        break

                    # 重新计算最小距离（使用最新点差）
                    new_stops_level = stops_level
                    if new_stops_level <= 0:
                        new_stops_level = max(10, min(50, int(current_spread_points * 5)))

                    # 应用更大的安全边际（重试时增加50%）
                    retry_safety_margin = 1.5 if retry_count == 0 else 2.0
                    retry_slippage_buffer = 50  # 重试时增加滑点缓冲
                    new_effective_stops_level = int(new_stops_level * retry_safety_margin) + retry_slippage_buffer

                    logger.info(
                        f"🔄 重新计算: 最新点差={current_spread_points:.1f}点, 新安全距离={new_effective_stops_level}点 (安全边际={retry_safety_margin:.0%})")

                    # 使用实际入场价格重新计算止盈止损
                    actual_entry = actual_position.price_open

                    # 重新计算止损
                    if sl_price > 0:
                        if signal['direction'] == 'BUY':
                            sl_price = actual_entry - new_effective_stops_level * point
                        else:  # SELL
                            sl_price = actual_entry + new_effective_stops_level * point
                        sl_price = self.normalize_price(sl_price, digits)

                    # 重新计算止盈
                    if tp_price > 0:
                        if signal['direction'] == 'BUY':
                            tp_price = actual_entry + new_effective_stops_level * point
                        else:  # SELL
                            tp_price = actual_entry - new_effective_stops_level * point
                        tp_price = self.normalize_price(tp_price, digits)

                    logger.info(f"🔄 重新计算的止盈止损: SL={sl_price:.{digits}f}, TP={tp_price:.{digits}f}")

                    # 更新请求
                    modify_request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": position_ticket,
                    }
                    if sl_price > 0:
                        modify_request["sl"] = sl_price
                    if tp_price > 0:
                        modify_request["tp"] = tp_price

                    # 重试
                    modify_result = mt5.order_send(modify_request)
                    retry_count += 1
                else:
                    # 重试次数用完，记录最终错误
                    if not setup_success:
                        logger.error(
                            f"❌ 止盈止损设置失败，已重试{max_retries}次，放弃设置。订单号: {order_ticket}, 持仓号: {position_ticket}")
                        # 即使失败也继续，不阻止开仓成功
                        break

            if not setup_success:
                logger.warning(f"⚠️ 止盈止损设置未成功，但开仓已完成。订单号: {order_ticket}, 持仓号: {position_ticket}")
                logger.warning(f"   建议手动检查并设置止盈止损: SL={sl_price:.{digits}f}, TP={tp_price:.{digits}f}")
            else:
                # 如果使用动态止盈，标记该订单
                if use_dynamic_tp and position_ticket:
                    self.dynamic_tp_positions.add(position_ticket)
                    self.last_dynamic_tp_update[position_ticket] = time.time()
                    logger.info(f"📈 订单{position_ticket}标记为动态止盈，将根据趋势强度实时调整止盈")

            self.daily_trades += 1

            return order_ticket

        except Exception as e:

            logger.error(f"开仓异常: {str(e)}")

            traceback.print_exc()

            return None

    def _detect_trend_exhaustion(self, ticket: int, position: Dict, indicators: Dict) -> bool:
        """检测趋势是否即将结束（使用净盈利）"""

        if not ProfessionalComplexConfig.TREND_EXHAUSTION['ENABLE']:
            return False

        try:
            # 使用净盈利（扣除手续费和点差）
            current_profit_usd, current_profit_pct = self._calculate_net_profit(position)

            # 检查是否达到最小净盈利要求
            if current_profit_pct < ProfessionalComplexConfig.TREND_EXHAUSTION['MIN_PROFIT_PCT']:
                return False

            position_type = position['type']
            entry_price = position['price_open']
            current_price = position.get('price_current', 0)

            if current_price <= 0:
                return False

            # 1. ADX下降（趋势强度减弱）
            current_adx = indicators.get('ADX', 0)
            previous_adx = indicators.get('ADX_PREV', 0)
            adx_declining = current_adx < previous_adx * 0.9 and current_adx < 25

            # 2. MACD背离检测
            macd_hist = indicators.get('MACD_HIST', 0)
            previous_macd_hist = indicators.get('MACD_HIST_PREV', 0)

            if position_type == 'BUY':
                price_rising = current_price > entry_price * 1.01
                macd_divergence = price_rising and macd_hist < previous_macd_hist * 0.8
            else:  # SELL
                price_falling = current_price < entry_price * 0.99
                macd_divergence = price_falling and macd_hist > previous_macd_hist * 0.8

            # 3. RSI超买/超卖后回落
            rsi = indicators.get('RSI_14', 50)
            if position_type == 'BUY':
                rsi_exhaustion = rsi > 70 and rsi < 65
            else:
                rsi_exhaustion = rsi < 30 and rsi > 35

            # 4. 价格动量减弱
            prices = list(self.data_engine.price_buffer)
            momentum_weakening = False
            if len(prices) >= 10:
                recent_momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
                previous_momentum = (prices[-5] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0

                if position_type == 'BUY':
                    momentum_weakening = recent_momentum < previous_momentum * 0.5 and recent_momentum < 0.0005
                else:
                    momentum_weakening = recent_momentum > previous_momentum * 0.5 and recent_momentum > -0.0005

            # 综合判断：至少2个信号同时出现
            exhaustion_signals = sum([
                adx_declining,
                macd_divergence,
                rsi_exhaustion,
                momentum_weakening
            ])

            if exhaustion_signals >= ProfessionalComplexConfig.TREND_EXHAUSTION['SIGNALS_REQUIRED']:
                logger.info(f"⚠️ 订单{ticket}检测到趋势衰竭信号: ADX下降={adx_declining}, "
                            f"MACD背离={macd_divergence}, RSI衰竭={rsi_exhaustion}, "
                            f"动量减弱={momentum_weakening}")
                return True

            return False

        except Exception as e:
            logger.warning(f"趋势衰竭检测异常: {str(e)}")
            return False

    def _update_dynamic_take_profit(self, ticket: int, position: Dict, indicators: Dict):
        """动态调整止盈 - 根据趋势强度和盈利情况，最大化止盈目标并实时更新控制回撤"""

        if not ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['ENABLE']:
            return

        if ticket not in self.dynamic_tp_positions:
            return

        try:
            current_price = position.get('price_current', 0)
            entry_price = position['price_open']
            position_type = position['type']
            current_tp = position.get('tp', 0)

            if current_price <= 0 or entry_price <= 0:
                return

            # 检查更新间隔
            current_time = time.time()
            last_update = self.last_dynamic_tp_update.get(ticket, 0)
            if current_time - last_update < ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['UPDATE_INTERVAL']:
                return

            # 计算当前净盈利（扣除手续费和点差）
            current_profit_usd, profit_pct = self._calculate_net_profit(position)

            # 至少达到最小净盈利才启用动态止盈
            if profit_pct < ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['MIN_PROFIT_PCT']:
                return

            # 获取趋势强度
            adx = indicators.get('ADX', 0)
            ema_alignment = abs(indicators.get('EMA_ALIGNMENT', 0))
            macd_trend = abs(indicators.get('MACD_TREND', 0))

            trend_strength = (adx / 50.0) * 0.4 + (ema_alignment) * 0.3 + (macd_trend) * 0.3
            trend_strength = min(1.0, trend_strength)

            # 动态止盈策略
            symbol_info = self.data_engine.data_validator.symbol_info
            if not symbol_info:
                return

            point = symbol_info.point
            digits = symbol_info.digits
            atr = indicators.get('ATR', 0)

            if atr <= 0:
                return

            # 根据趋势强度和盈利情况调整止盈
            if trend_strength > 0.7:
                # 强趋势：使用更大的ATR倍数，最大化止盈
                tp_distance = atr * ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['STRONG_TREND_ATR_MULT']
            elif trend_strength > 0.5:
                # 中等趋势：标准ATR倍数
                tp_distance = atr * ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['MEDIUM_TREND_ATR_MULT']
            else:
                # 弱趋势：收紧止盈
                tp_distance = atr * ProfessionalComplexConfig.DYNAMIC_TAKE_PROFIT['WEAK_TREND_ATR_MULT']

            # 计算新止盈价
            if position_type == 'BUY':
                new_tp = current_price + tp_distance
            else:
                new_tp = current_price - tp_distance

            # 确保新止盈价优于当前止盈价（更有利）
            if position_type == 'BUY':
                if new_tp > current_tp or current_tp == 0:
                    new_tp = self.normalize_price(new_tp, digits)
                    self._update_take_profit(ticket, new_tp)
                    self.last_dynamic_tp_update[ticket] = current_time
                    logger.info(f"📈 订单{ticket}动态止盈更新: {current_tp:.{digits}f} -> {new_tp:.{digits}f} "
                                f"(趋势强度={trend_strength:.2f}, 盈利={profit_pct:.2%}, ATR倍数={tp_distance / atr:.2f})")
            else:  # SELL
                if new_tp < current_tp or current_tp == 0:
                    new_tp = self.normalize_price(new_tp, digits)
                    self._update_take_profit(ticket, new_tp)
                    self.last_dynamic_tp_update[ticket] = current_time
                    logger.info(f"📈 订单{ticket}动态止盈更新: {current_tp:.{digits}f} -> {new_tp:.{digits}f} "
                                f"(趋势强度={trend_strength:.2f}, 盈利={profit_pct:.2%}, ATR倍数={tp_distance / atr:.2f})")

        except Exception as e:
            logger.warning(f"动态止盈更新异常: {str(e)}")

    def _monitor_profit_drawdown(self, ticket: int, position: Dict):
        """监控单笔订单的盈利回撤（基于价格差：盈利价与入场价）"""

        if not ProfessionalComplexConfig.PROFIT_DRAWDOWN_CONTROL['ENABLE']:
            return

        try:
            current_price = position.get('price_current', 0)
            entry_price = position['price_open']
            position_type = position['type']
            volume = position.get('volume', 0)

            if current_price <= 0 or entry_price <= 0:
                return

            # 计算价格差（美元，不受手数影响）
            # 价格差直接等于美元数，不需要转换
            # 例如：价格从2000.00涨到2000.08，价格差=0.08，对应0.08美元
            # 例如：价格从2000.00涨到2008.00，价格差=8.00，对应8美元
            if position_type == 'BUY':
                price_diff_usd = current_price - entry_price
            else:  # SELL
                price_diff_usd = entry_price - current_price

            # 如果价格差未达到8美元，不启用回撤保护
            if price_diff_usd < ProfessionalComplexConfig.PROFIT_DRAWDOWN_CONTROL['MIN_PEAK_PROFIT_USD']:
                # 清理记录（如果之前有记录）
                if ticket in self.position_peak_profit:
                    del self.position_peak_profit[ticket]
                return

            # 初始化或更新峰值价格差
            if ticket not in self.position_peak_profit:
                # 首次达到8美元，立即记录为峰值
                self.position_peak_profit[ticket] = {
                    'peak_profit_usd': price_diff_usd
                }
                logger.info(
                    f"📊 订单{ticket}价格差达到${price_diff_usd:.2f}，开始监控回撤 (入场价={entry_price:.2f}, 当前价={current_price:.2f})")
            else:
                peak_info = self.position_peak_profit[ticket]
                # 更新峰值（价格差）
                if price_diff_usd > peak_info['peak_profit_usd']:
                    peak_info['peak_profit_usd'] = price_diff_usd
                    logger.info(
                        f"📊 订单{ticket}价格差创新高: ${price_diff_usd:.2f} (入场价={entry_price:.2f}, 当前价={current_price:.2f})")

            peak_info = self.position_peak_profit[ticket]
            peak_profit_usd = peak_info['peak_profit_usd']

            # 计算盈利回撤（美元回撤）
            # 只要当前价格差小于峰值，就计算回撤
            if price_diff_usd < peak_profit_usd:
                profit_drawdown_usd = peak_profit_usd - price_diff_usd

                # 如果盈利回撤达到或超过3美元，触发保护性平仓
                if profit_drawdown_usd >= ProfessionalComplexConfig.PROFIT_DRAWDOWN_CONTROL['MAX_DRAWDOWN_USD']:
                    logger.warning(f"⚠️ 订单{ticket}价格差回撤超限: "
                                   f"峰值=${peak_profit_usd:.2f}, 当前=${price_diff_usd:.2f}, "
                                   f"回撤=${profit_drawdown_usd:.2f} >= ${ProfessionalComplexConfig.PROFIT_DRAWDOWN_CONTROL['MAX_DRAWDOWN_USD']:.2f}")

                    # 保护性平仓
                    self._close_position(ticket, position_type)
                    logger.info(
                        f"🛡️ 订单{ticket}因价格差回撤${profit_drawdown_usd:.2f}（超过${ProfessionalComplexConfig.PROFIT_DRAWDOWN_CONTROL['MAX_DRAWDOWN_USD']:.2f}）已平仓")

                    # 清理记录
                    if ticket in self.position_peak_profit:
                        del self.position_peak_profit[ticket]
                    if ticket in self.dynamic_tp_positions:
                        self.dynamic_tp_positions.remove(ticket)
                    return  # 平仓后直接返回

        except Exception as e:
            logger.warning(f"监控盈利回撤异常: {str(e)}")

    def update_positions(self):

        """更新持仓状态（跟踪止损、多目标止盈、动态止盈、盈利回撤监控等）"""

        try:

            positions = self.get_open_positions()

            indicators = self.data_engine.calculate_complex_indicators()

            current_price = indicators.get('CURRENT_PRICE', 0)

            if not current_price:
                return

            for ticket, pos in positions.items():

                # 1. 监控盈利回撤（优先检查，保护已有盈利）
                self._monitor_profit_drawdown(ticket, pos)

                # 2. 检查多目标止盈（非动态止盈订单）
                if ticket not in self.dynamic_tp_positions:
                    self._check_multi_target_take_profit(ticket, pos, current_price)

                # 3. 检测趋势衰竭（如果净盈利足够）
                # 使用净盈利检查（_detect_trend_exhaustion内部会检查净盈利）
                if self._detect_trend_exhaustion(ticket, pos, indicators):
                    # 趋势衰竭时，如果是动态止盈订单，收紧止盈
                    if ticket in self.dynamic_tp_positions:
                        logger.info(f"⚠️ 订单{ticket}趋势衰竭，收紧动态止盈")
                        # 动态止盈会在下次更新时自动收紧

                # 4. 动态止盈更新（单边明确趋势，最大化止盈目标并实时更新控制回撤）
                self._update_dynamic_take_profit(ticket, pos, indicators)

                # 5. 更新跟踪止损

                if ProfessionalComplexConfig.RISK_MANAGEMENT['STOP_LOSS']['TRAILING']['ACTIVATION_PERCENT'] > 0:
                    self._update_trailing_stop(ticket, pos, current_price)

        except Exception as e:

            logger.error(f"更新持仓异常: {str(e)}")

    def _update_trailing_stop(self, ticket: int, position: Dict, current_price: float):

        """更新跟踪止损"""

        try:

            trailing_config = ProfessionalComplexConfig.RISK_MANAGEMENT['STOP_LOSS']['TRAILING']

            activation_percent = trailing_config['ACTIVATION_PERCENT']

            step_size = trailing_config['STEP_SIZE']

            entry_price = position['price_open']

            if entry_price <= 0 or current_price <= 0:
                return

            current_sl = position.get('sl', 0)

            if position['type'] == 'BUY':

                profit_percent = (current_price - entry_price) / entry_price

                if profit_percent >= activation_percent:

                    # 计算新的止损价格（使用点数而不是百分比）
                    symbol_info = self.data_engine.data_validator.symbol_info
                    if symbol_info:
                        point = symbol_info.point
                        # step_size是百分比，转换为点数
                        step_points = step_size * entry_price / point
                        new_sl = current_price - step_points * point

                        # 确保新止损高于当前止损（或当前没有止损）
                        if new_sl > current_sl or current_sl == 0:
                            # 确保新止损不会高于入场价
                            if new_sl < entry_price:
                                self._modify_stop_loss(ticket, new_sl)

            else:  # SELL

                profit_percent = (entry_price - current_price) / entry_price

                if profit_percent >= activation_percent:

                    # 计算新的止损价格
                    symbol_info = self.data_engine.data_validator.symbol_info
                    if symbol_info:
                        point = symbol_info.point
                        # step_size是百分比，转换为点数
                        step_points = step_size * entry_price / point
                        new_sl = current_price + step_points * point

                        # 确保新止损低于当前止损（或当前没有止损）
                        if new_sl < current_sl or current_sl == 0:
                            # 确保新止损不会低于入场价
                            if new_sl > entry_price:
                                self._modify_stop_loss(ticket, new_sl)

        except Exception as e:

            logger.debug(f"更新跟踪止损异常: {str(e)}")

    def _modify_stop_loss(self, ticket: int, new_sl: float):

        """修改止损"""

        try:
            symbol = self.data_engine.symbol
            symbol_info = self.data_engine.data_validator.symbol_info

            if not symbol_info:
                logger.warning(f"⚠️ 无法获取品种信息，跳过修改止损")
                return

            # 规范化价格
            digits = symbol_info.digits
            new_sl = self.normalize_price(new_sl, digits)

            # 获取当前持仓信息以验证止损价格
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for pos in positions:
                    if pos.ticket == ticket:
                        entry_price = pos.price_open
                        position_type = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'

                        # 验证止损价格方向
                        if position_type == 'BUY' and new_sl >= entry_price:
                            logger.warning(f"⚠️ 止损价格无效（BUY订单止损应低于入场价），跳过修改")
                            return
                        elif position_type == 'SELL' and new_sl <= entry_price:
                            logger.warning(f"⚠️ 止损价格无效（SELL订单止损应高于入场价），跳过修改")
                            return
                        break

            request = {

                "action": mt5.TRADE_ACTION_SLTP,

                "symbol": symbol,

                "position": ticket,

                "sl": new_sl,

            }

            result = mt5.order_send(request)

            if result is None:
                error_code = mt5.last_error()
                logger.warning(f"⚠️ 修改止损失败: order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")
                return

            if result.retcode == mt5.TRADE_RETCODE_DONE:

                logger.debug(f"✅ 止损已更新: {ticket} -> {new_sl:.{digits}f}")

            else:

                logger.warning(f"⚠️ 修改止损失败: {result.retcode} - {result.comment}")

        except Exception as e:

            logger.warning(f"修改止损异常: {str(e)}")

    def _check_multi_target_take_profit(self, ticket: int, position: Dict, current_price: float):

        """检查多目标止盈并执行部分平仓"""

        try:

            # 检查是否有该持仓的多目标止盈信息

            if ticket not in self.position_tp_targets:
                return

            tp_targets = self.position_tp_targets[ticket]

            if not tp_targets or len(tp_targets) == 0:
                return

            position_type = position['type']

            entry_price = position['price_open']

            current_volume = position['volume']

            # 检查每个止盈目标

            for i, tp_target in enumerate(tp_targets):

                tp_price = tp_target['price']

                close_percent = tp_target['close_percent']

                # 检查是否达到止盈目标

                target_reached = False

                if position_type == 'BUY':

                    # BUY订单：当前价格 >= 止盈价格

                    if current_price >= tp_price:
                        target_reached = True

                else:  # SELL

                    # SELL订单：当前价格 <= 止盈价格

                    if current_price <= tp_price:
                        target_reached = True

                if target_reached:

                    # 重新获取最新持仓信息（可能已经被部分平仓）
                    positions = self.get_open_positions()
                    if ticket not in positions:
                        # 持仓已被完全平仓，清理多目标止盈信息
                        if ticket in self.position_tp_targets:
                            del self.position_tp_targets[ticket]
                        return

                    # 使用最新持仓信息
                    latest_position = positions[ticket]
                    current_volume = latest_position['volume']

                    # 计算需要平仓的手数

                    close_volume = current_volume * close_percent

                    # 确保最小手数

                    symbol_info = self.data_engine.data_validator.symbol_info

                    if symbol_info:
                        min_lot = symbol_info.volume_min

                        lot_step = symbol_info.volume_step

                        # 四舍五入到步长

                        close_volume = round(close_volume / lot_step) * lot_step

                        close_volume = max(min_lot, close_volume)

                    # 确保不超过当前持仓

                    if close_volume >= current_volume:
                        close_volume = current_volume

                    # 执行部分平仓

                    if close_volume > 0:

                        success = self._partial_close_position(ticket, close_volume, position_type)

                        if success:

                            logger.info(
                                f"🎯 达到止盈目标TP{i + 1} ({tp_price:.{symbol_info.digits if symbol_info else 2}f})，部分平仓: {close_volume}手")

                            # 等待部分平仓完成
                            time.sleep(0.3)

                            # 从目标列表中移除已触发的目标

                            tp_targets.pop(i)

                            # 更新剩余持仓的止盈目标

                            if len(tp_targets) > 0:

                                # 更新MT5的止盈价格为下一个目标

                                next_tp = tp_targets[0]['price']

                                self._update_take_profit(ticket, next_tp)

                            else:

                                # 所有目标都已完成，移除该持仓的多目标止盈信息

                                del self.position_tp_targets[ticket]

                            break  # 一次只处理一个目标
                        else:
                            logger.warning(f"⚠️ 部分平仓失败，无法执行止盈目标TP{i + 1}")
                            break  # 平仓失败，等待下次检查

        except Exception as e:

            logger.error(f"检查多目标止盈异常: {str(e)}")

            traceback.print_exc()

    def _close_position(self, ticket: int, position_type: str) -> bool:

        """完全平仓"""

        try:

            symbol = self.data_engine.symbol

            symbol_info = self.data_engine.data_validator.symbol_info

            if not symbol_info:
                return False

            # 获取持仓信息以获取手数

            positions = mt5.positions_get(symbol=symbol)

            if not positions:
                logger.warning(f"⚠️ 未找到持仓 {ticket}")

                return False

            position = None

            for pos in positions:

                if pos.ticket == ticket:
                    position = pos

                    break

            if not position:
                logger.warning(f"⚠️ 未找到持仓 {ticket}")

                return False

            volume = position.volume

            # 获取当前价格

            tick = mt5.symbol_info_tick(symbol)

            if not tick:
                return False

            ask = DataSourceValidator._get_tick_value(tick, 'ask')

            bid = DataSourceValidator._get_tick_value(tick, 'bid')

            # 确定平仓价格和类型

            if position_type == 'BUY':

                close_price = bid  # BUY订单用bid价平仓

                close_type = mt5.ORDER_TYPE_SELL  # 卖出平仓

            else:

                close_price = ask  # SELL订单用ask价平仓

                close_type = mt5.ORDER_TYPE_BUY  # 买入平仓

            # 发送平仓订单

            request = {

                "action": mt5.TRADE_ACTION_DEAL,

                "symbol": symbol,

                "volume": volume,

                "type": close_type,

                "position": ticket,  # 指定要平仓的持仓ticket

                "price": close_price,

                "deviation": 20,

                "magic": 123456,

                "comment": f"Close_Reversal",

                "type_time": mt5.ORDER_TIME_GTC,

            }

            result = mt5.order_send(request)

            if result is None:
                error_code = mt5.last_error()

                logger.warning(f"⚠️ 平仓失败: order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")

                return False

            if result.retcode == mt5.TRADE_RETCODE_DONE:

                logger.info(f"✅ 平仓成功: {volume}手 @ {close_price:.2f} (ticket: {ticket})")

                # 清理相关记录

                if ticket in self.position_tp_targets:
                    del self.position_tp_targets[ticket]

                if ticket in self.sl_tp_set_positions:
                    self.sl_tp_set_positions.discard(ticket)

                # 清理动态止盈和盈利回撤记录
                if ticket in self.dynamic_tp_positions:
                    self.dynamic_tp_positions.discard(ticket)

                if ticket in self.last_dynamic_tp_update:
                    del self.last_dynamic_tp_update[ticket]

                if ticket in self.position_peak_profit:
                    del self.position_peak_profit[ticket]

                # 记录信号结果用于ML训练（如果存在信号特征）
                if hasattr(self, 'position_signal_features') and ticket in self.position_signal_features:
                    signal_features = self.position_signal_features[ticket]
                    entry_price = position.price_open
                    # MT5 TradePosition对象使用time属性，不是time_open
                    entry_time = position.time if hasattr(position, 'time') else 0
                    hold_duration = time.time() - entry_time if entry_time > 0 else 0

                    # 计算盈亏
                    if position_type == 'BUY':
                        profit_usd = (close_price - entry_price) * volume * 100
                        was_profitable = close_price > entry_price
                    else:  # SELL
                        profit_usd = (entry_price - close_price) * volume * 100
                        was_profitable = close_price < entry_price

                    # 记录到ML评估器（如果可用）
                    # 注意：需要通过策略主类访问signal_generator
                    # 这里先记录，策略主类会定期处理
                    if hasattr(self, 'signal_generator') and hasattr(self.signal_generator, 'ml_evaluator'):
                        try:
                            self.signal_generator.ml_evaluator.record_signal_outcome(
                                signal_features, was_profitable, profit_usd, hold_duration
                            )
                        except Exception as e:
                            logger.warning(f"记录信号结果异常: {str(e)}")

                    # 清理信号特征记录
                    del self.position_signal_features[ticket]

                return True

            else:

                logger.warning(f"⚠️ 平仓失败: {result.retcode} - {result.comment}")

                return False

        except Exception as e:

            logger.error(f"平仓异常: {str(e)}")

            traceback.print_exc()

            return False

    def _partial_close_position(self, ticket: int, volume: float, position_type: str) -> bool:

        """部分平仓"""

        try:

            symbol = self.data_engine.symbol

            symbol_info = self.data_engine.data_validator.symbol_info

            if not symbol_info:
                return False

            # 获取当前价格

            tick = mt5.symbol_info_tick(symbol)

            if not tick:
                return False

            ask = DataSourceValidator._get_tick_value(tick, 'ask')

            bid = DataSourceValidator._get_tick_value(tick, 'bid')

            # 确定平仓价格和类型

            if position_type == 'BUY':

                close_price = bid  # BUY订单用bid价平仓

                close_type = mt5.ORDER_TYPE_SELL  # 卖出平仓

            else:

                close_price = ask  # SELL订单用ask价平仓

                close_type = mt5.ORDER_TYPE_BUY  # 买入平仓

            # 发送平仓订单

            request = {

                "action": mt5.TRADE_ACTION_DEAL,

                "symbol": symbol,

                "volume": volume,

                "type": close_type,

                "position": ticket,  # 指定要平仓的持仓ticket

                "price": close_price,

                "deviation": 20,

                "magic": 123456,

                "comment": f"Partial_Close_TP",

                "type_time": mt5.ORDER_TIME_GTC,

            }

            result = mt5.order_send(request)

            if result is None:
                error_code = mt5.last_error()

                logger.warning(f"⚠️ 部分平仓失败: order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")

                return False

            if result.retcode == mt5.TRADE_RETCODE_DONE:

                logger.info(f"✅ 部分平仓成功: {volume}手 @ {close_price:.2f}")

                return True

            else:

                logger.warning(f"⚠️ 部分平仓失败: {result.retcode} - {result.comment}")

                return False

        except Exception as e:

            logger.error(f"部分平仓异常: {str(e)}")

            traceback.print_exc()

            return False

    def _update_take_profit(self, ticket: int, new_tp: float):

        """更新止盈价格"""

        try:

            symbol_info = self.data_engine.data_validator.symbol_info

            if not symbol_info:
                return

            # 获取当前持仓信息
            positions = mt5.positions_get(symbol=self.data_engine.symbol)
            if not positions:
                logger.warning(f"⚠️ 未找到持仓 {ticket}")
                return

            position = None
            for pos in positions:
                if pos.ticket == ticket:
                    position = pos
                    break

            if not position:
                logger.warning(f"⚠️ 未找到持仓 {ticket}")
                return

            # 获取最小止损距离
            point = symbol_info.point
            digits = symbol_info.digits
            stops_level = 0
            try:
                if hasattr(symbol_info, 'trade_stops_level'):
                    stops_level = symbol_info.trade_stops_level
                elif hasattr(symbol_info, 'stops_level'):
                    stops_level = symbol_info.stops_level
            except:
                pass

            if stops_level <= 0:
                current_spread = (symbol_info.ask - symbol_info.bid) / point
                stops_level = max(10, min(50, int(current_spread * 5)))

            # 验证止盈价格
            entry_price = position.price_open
            position_type = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

            # 计算止盈距离
            if position_type == 'BUY':
                tp_distance = (new_tp - entry_price) / point
                # BUY订单：止盈应高于入场价，且距离至少为stops_level
                if new_tp <= entry_price:
                    logger.warning(f"⚠️ 止盈价格无效（BUY订单止盈应高于入场价 {entry_price:.{digits}f}），跳过更新")
                    return
                if tp_distance < stops_level:
                    # 调整止盈价格
                    new_tp = entry_price + stops_level * point
                    new_tp = self.normalize_price(new_tp, digits)
                    logger.debug(f"调整止盈价格以满足最小距离要求: {stops_level}点")
            else:  # SELL
                tp_distance = (entry_price - new_tp) / point
                # SELL订单：止盈应低于入场价，且距离至少为stops_level
                if new_tp >= entry_price:
                    logger.warning(f"⚠️ 止盈价格无效（SELL订单止盈应低于入场价 {entry_price:.{digits}f}），跳过更新")
                    return
                if tp_distance < stops_level:
                    # 调整止盈价格
                    new_tp = entry_price - stops_level * point
                    new_tp = self.normalize_price(new_tp, digits)
                    logger.debug(f"调整止盈价格以满足最小距离要求: {stops_level}点")

            # 规范化止盈价格
            new_tp = self.normalize_price(new_tp, digits)

            request = {

                "action": mt5.TRADE_ACTION_SLTP,

                "symbol": self.data_engine.symbol,

                "position": ticket,

                "tp": new_tp,

            }

            result = mt5.order_send(request)

            if result is None:
                error_code = mt5.last_error()
                logger.warning(f"⚠️ 更新止盈价格失败: order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")
                return

            if result.retcode == mt5.TRADE_RETCODE_DONE:

                logger.debug(f"✅ 止盈价格已更新: {ticket} -> {new_tp:.2f}")

            else:

                logger.warning(f"⚠️ 更新止盈价格失败: {result.retcode} - {result.comment}")

        except Exception as e:

            logger.error(f"更新止盈价格异常: {str(e)}")
            traceback.print_exc()


class ProfessionalComplexStrategy:
    """专业复杂策略主类 - 整合所有组件"""

    def __init__(self, validator: DataSourceValidator):

        self.validator = validator

        self.data_engine = ProfessionalTickDataEngine(validator)

        self.market_analyzer = AdvancedMarketStateAnalyzer(self.data_engine)

        self.signal_generator = ProfessionalSignalGenerator(self.data_engine, self.market_analyzer)

        self.risk_manager = ComplexRiskManager(self.data_engine)

        self.position_manager = ProfessionalPositionManager(self.data_engine, self.risk_manager)

        # 设置signal_generator引用到position_manager（用于ML训练）
        self.position_manager.signal_generator = self.signal_generator

        self.running = False

        self.processing_thread = None

        # ML模型训练相关
        self.last_ml_training_time = 0
        self.ml_training_interval = 3600  # 每1小时训练一次（如果样本足够）

    def run_strategy(self):

        """运行策略主循环"""

        logger.info("🚀 启动策略主循环...")

        self.running = True

        try:

            # 数据收集阶段

            logger.info("📊 数据收集阶段...")

            collection_start = time.time()

            while time.time() - collection_start < 30:  # 收集30秒数据

                self.data_engine.process_tick_data()

                time.sleep(ProfessionalComplexConfig.PROCESSING_INTERVAL)

            if not self.data_engine.initialized:
                logger.error("❌ 数据引擎初始化失败")

                return

            logger.info("✅ 数据收集完成，开始交易循环...")

            # 主交易循环

            last_analysis_time = 0
            last_heartbeat_time = time.time()
            heartbeat_interval = 10.0  # 每10秒输出一次心跳日志
            last_diagnostic_time = 0
            diagnostic_interval = 30.0  # 每30秒输出一次诊断信息

            analysis_interval = 1.0  # 每秒分析一次

            logger.info("🔄 进入主交易循环，开始处理数据...")

            # 立即执行一次市场状态分析，显示初始状态
            try:
                market_state, state_confidence = self.market_analyzer.analyze_complex_market_state()
                indicators = self.data_engine.calculate_complex_indicators()
                if indicators:
                    current_price = indicators.get('CURRENT_PRICE', 0)
                    # 显示所有状态的原始概率
                    raw_probs = {
                        'TRENDING': self.market_analyzer._calculate_trending_probability(indicators),
                        'RANGING': self.market_analyzer._calculate_ranging_probability(indicators),
                        'VOLATILE': self.market_analyzer._calculate_volatile_probability(indicators),
                    }
                    logger.info(
                        f"📊 初始市场状态: {market_state} (置信度: {state_confidence:.2f}), 当前价格: {current_price:.2f}")
                    logger.info(f"   原始概率: TRENDING={raw_probs['TRENDING']:.3f}, "
                                f"RANGING={raw_probs['RANGING']:.3f}, "
                                f"VOLATILE={raw_probs['VOLATILE']:.3f}")
                else:
                    logger.warning(f"⚠️ 初始状态分析: 无法计算技术指标")
            except Exception as init_error:
                logger.warning(f"⚠️ 初始状态分析异常: {str(init_error)}")
                traceback.print_exc()

            while self.running:

                try:

                    current_time = time.time()

                    # 心跳日志（每10秒输出一次，确认程序在运行）
                    if current_time - last_heartbeat_time >= heartbeat_interval:
                        tick_count = len(self.data_engine.tick_buffer)
                        logger.info(
                            f"💓 程序运行中... Tick缓冲区: {tick_count}个, 数据引擎已初始化: {self.data_engine.initialized}")
                        last_heartbeat_time = current_time

                    # 处理Tick数据

                    tick_result = self.data_engine.process_tick_data()
                    if not tick_result:
                        # 如果处理失败，等待一下再继续
                        time.sleep(ProfessionalComplexConfig.PROCESSING_INTERVAL)
                        continue

                    # 定期分析（降低频率）

                    if current_time - last_analysis_time >= analysis_interval:

                        try:
                            # 更新账户信息

                            self.risk_manager.update_account_info()

                            # 更新持仓状态

                            self.position_manager.update_positions()

                            # 生成交易信号

                            signal = self.signal_generator.generate_trading_signal()

                            if signal:

                                # 尝试开仓
                                logger.info(
                                    f"🔍 准备开仓: {signal.get('direction')} 强度: {signal.get('strength', 0):.2f} 价格: {signal.get('entry_price', 0):.2f}")
                                order_ticket = self.position_manager.open_position(signal)
                                if order_ticket:
                                    logger.info(f"✅ 开仓成功，订单号: {order_ticket}")
                                else:
                                    logger.info(f"⏸️ 开仓未执行（可能被can_open_new_position阻止）")
                            else:
                                # 如果没有信号，记录详细信息（降低频率）
                                if current_time - last_diagnostic_time >= diagnostic_interval:
                                    try:
                                        market_state, state_confidence = self.market_analyzer.analyze_complex_market_state()
                                        indicators = self.data_engine.calculate_complex_indicators()
                                        current_tick = self.data_engine.tick_buffer[
                                            -1] if self.data_engine.tick_buffer else None

                                        if indicators and current_tick:
                                            current_price = indicators.get('CURRENT_PRICE',
                                                                           current_tick.get('mid_price', 0))
                                            # 显示一些关键指标
                                            rsi_14 = indicators.get('RSI_14', 'N/A')
                                            adx = indicators.get('ADX', 'N/A')
                                            ema_alignment = indicators.get('EMA_ALIGNMENT', 'N/A')
                                            # 获取所有状态的原始概率用于诊断
                                            raw_probs = {
                                                'TRENDING': self.market_analyzer._calculate_trending_probability(
                                                    indicators),
                                                'RANGING': self.market_analyzer._calculate_ranging_probability(
                                                    indicators),
                                                'VOLATILE': self.market_analyzer._calculate_volatile_probability(
                                                    indicators),
                                            }
                                            logger.info(f"📊 市场状态: {market_state} (置信度: {state_confidence:.2f}), "
                                                        f"价格: {current_price:.2f}, "
                                                        f"RSI14: {rsi_14}, ADX: {adx}, EMA对齐: {ema_alignment}")
                                            logger.info(f"   原始概率: TRENDING={raw_probs['TRENDING']:.3f}, "
                                                        f"RANGING={raw_probs['RANGING']:.3f}, "
                                                        f"VOLATILE={raw_probs['VOLATILE']:.3f}, "
                                                        f"未生成交易信号")
                                        else:
                                            logger.warning(f"⚠️ 无法获取指标或Tick数据，无法生成信号")
                                        last_diagnostic_time = current_time
                                    except Exception as diag_error:
                                        logger.warning(f"⚠️ 诊断信息获取异常: {str(diag_error)}")
                                        last_diagnostic_time = current_time

                        except Exception as e:
                            logger.error(f"⚠️ 分析阶段异常: {str(e)}")
                            traceback.print_exc()

                        last_analysis_time = current_time

                        # 定期训练ML模型（如果样本足够）
                        if current_time - self.last_ml_training_time >= self.ml_training_interval:
                            try:
                                ml_report = self.signal_generator.ml_evaluator.get_evaluation_report()
                                training_samples = ml_report.get('training_samples', 0)

                                if training_samples >= 100:  # 至少100个样本才训练
                                    logger.info(f"🤖 开始定期ML模型训练，样本数: {training_samples}")
                                    if self.signal_generator.ml_evaluator.train_model(min_samples=100):
                                        # 训练成功后输出评估报告
                                        updated_report = self.signal_generator.ml_evaluator.get_evaluation_report()
                                        metrics = updated_report.get('metrics', {})
                                        logger.info(f"📊 ML模型评估报告:")
                                        logger.info(f"   准确率: {metrics.get('accuracy', 0):.2%}")
                                        logger.info(f"   精确率: {metrics.get('precision', 0):.2%}")
                                        logger.info(f"   召回率: {metrics.get('recall', 0):.2%}")
                                        logger.info(f"   F1分数: {metrics.get('f1_score', 0):.2%}")
                                        logger.info(f"   训练样本: {updated_report.get('training_samples', 0)}")
                                    else:
                                        logger.info(f"⏸️ ML模型训练未执行（样本不足或其他原因）")
                                else:
                                    if int(current_time) % 300 == 0:  # 每5分钟记录一次
                                        logger.info(f"📊 ML训练数据收集: {training_samples}/100 样本")

                                self.last_ml_training_time = current_time
                            except Exception as ml_error:
                                logger.warning(f"⚠️ ML训练异常: {str(ml_error)}")

                    time.sleep(ProfessionalComplexConfig.PROCESSING_INTERVAL)

                except KeyboardInterrupt:

                    logger.info("⚠️ 收到中断信号，停止策略...")

                    self.running = False

                    break

                except Exception as e:

                    logger.error(f"策略循环异常: {str(e)}")

                    traceback.print_exc()

                    time.sleep(1)

                    # 即使异常也继续运行，避免程序停止
                    logger.info("🔄 异常处理后继续运行...")

        except Exception as e:

            logger.error(f"策略运行异常: {str(e)}")

            traceback.print_exc()

        finally:

            self.stop_strategy()

    def stop_strategy(self):

        """停止策略"""

        logger.info("🛑 停止策略...")

        self.running = False

        # 打印最终统计

        positions = self.position_manager.get_open_positions()

        logger.info(f"当前持仓数: {len(positions)}")

        logger.info(f"今日交易数: {self.position_manager.daily_trades}")


def main():
    """主函数"""

    print("=" * 60)

    print("🎯 专业复杂策略（数据源修复版）")

    print("特点: 保持所有复杂性 + 修复数据源问题")

    print("包含: 多指标系统 + 市场状态识别 + 复杂信号生成")

    print("修复: 点差问题 + 数据质量验证 + 品种选择")

    print("=" * 60)

    # 初始化MT5连接

    if not mt5.initialize():
        logger.error("❌ MT5初始化失败")

        return

    try:

        # 登录账户（如果需要）

        if ProfessionalComplexConfig.LOGIN and ProfessionalComplexConfig.PASSWORD:

            authorized = mt5.login(

                login=ProfessionalComplexConfig.LOGIN,

                password=ProfessionalComplexConfig.PASSWORD,

                server=ProfessionalComplexConfig.SERVER

            )

            if not authorized:

                logger.warning(f"⚠️ 账户登录失败: {mt5.last_error()}")

                logger.info("继续使用当前连接...")

            else:

                account_info = mt5.account_info()

                if account_info:
                    logger.info(f"✅ 账户连接成功: {account_info.login} | "

                                f"余额: {account_info.balance:.2f} | "

                                f"服务器: {account_info.server}")

        # 检查自动交易状态
        terminal_info = mt5.terminal_info()
        if terminal_info:
            if not terminal_info.trade_allowed:
                logger.warning(f"⚠️ MT5自动交易可能被禁用")
                logger.warning(f"💡 请在MT5客户端中启用自动交易:")
                logger.warning(f"   1. 点击工具栏上的'自动交易'按钮（或按 Ctrl+E）")
                logger.warning(f"   2. 确保按钮显示为绿色（已启用）")
                logger.warning(f"   3. 或者：工具 -> 选项 -> 专家顾问 -> 勾选'允许自动交易'")
            else:
                logger.info(f"✅ MT5自动交易已启用")

        # 寻找有效品种
        logger.info("🔍 开始寻找有效交易品种...")
        validator = DataSourceValidator()
        valid_symbol = validator.find_valid_symbol()

        if not valid_symbol:
            logger.error("❌ 未找到有效交易品种")
            mt5.shutdown()
            return

        # 显示品种信息
        symbol_info = validator.get_symbol_info()

        if symbol_info:
            logger.info(f"📊 品种信息:")

            logger.info(f"   名称: {symbol_info['name']}")

            logger.info(f"   当前价格: {symbol_info['bid']:.2f} / {symbol_info['ask']:.2f}")

            logger.info(f"   点差: {symbol_info['spread']:.1f}点")

            logger.info(f"   精度: {symbol_info['digits']}位")

        # 初始化并运行策略

        logger.info("🚀 初始化策略组件...")

        strategy = ProfessionalComplexStrategy(validator)

        logger.info("=" * 60)

        logger.info("✅ 所有组件初始化完成")

        logger.info("📈 开始运行策略...")

        logger.info("💡 按 Ctrl+C 停止策略")

        logger.info("=" * 60)

        # 运行策略

        strategy.run_strategy()

    except KeyboardInterrupt:

        logger.info("\n⚠️ 用户中断，正在关闭...")

    except Exception as e:

        logger.error(f"❌ 主程序异常: {str(e)}")

        traceback.print_exc()

    finally:

        logger.info("🛑 关闭MT5连接...")

        mt5.shutdown()

        logger.info("✅ 程序已退出")


if __name__ == "__main__":
    main()
