import MetaTrader5 as mt5

import pandas as pd

import numpy as np

import time

import logging

from datetime import datetime, timedelta

import sys

import talib

from collections import deque

import math

import threading

from typing import Dict, List, Tuple, Optional, Any

import traceback

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

            'PERIODS': [5, 10, 20, 50, 100]

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

        'MIN_STRENGTH': 0.5,

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

            # 获取历史Tick数据

            end_time = datetime.now()

            start_time = end_time - timedelta(minutes=5)

            ticks = mt5.copy_ticks_range(symbol, start_time, end_time, mt5.COPY_TICKS_ALL)

            # 修复numpy数组判断问题

            if ticks is None:
                logger.warning(f"  无法获取Tick数据")

                return False

            # 使用size属性检查numpy数组

            if hasattr(ticks, 'size'):

                if ticks.size == 0 or ticks.size < 10:
                    logger.warning(f"  数据不足: {ticks.size}个Tick")

                    return False

                ticks_len = ticks.size

            else:

                # 如果是列表或其他类型

                ticks_len = len(ticks) if ticks else 0

                if ticks_len < 10:
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

            # 如果历史数据不足，尝试使用实时tick

            if valid_ticks < 5:
                logger.warning(f"  历史Tick数据不足: {valid_ticks}个有效Tick")

                logger.info(f"  尝试使用实时Tick数据验证...")

                return self._test_realtime_tick_quality(symbol, symbol_info)

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

            for i in range(10):  # 尝试获取10个实时tick

                tick = mt5.symbol_info_tick(symbol)

                if tick:

                    ask = self._get_tick_value(tick, 'ask')

                    bid = self._get_tick_value(tick, 'bid')

                    # 只检查价格有效性，不检查点差

                    if ask > bid > 0:
                        spread = (ask - bid) * 10000

                        spreads.append(spread)

                        valid_samples += 1

                time.sleep(0.1)  # 等待0.1秒获取下一个tick

            if valid_samples >= 3:  # 至少需要3个有效样本

                avg_spread = np.mean(spreads) if spreads else 0

                logger.info(f"  实时Tick验证成功: {valid_samples}个有效样本, 平均点差: {avg_spread:.1f}点")

                self.symbol_info = symbol_info

                self.connection_quality['avg_spread'] = avg_spread

                self.connection_quality['success_rate'] = valid_samples / 10.0

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

    def process_tick_data(self) -> bool:

        """处理Tick数据 - 增强容错"""

        try:

            tick = mt5.symbol_info_tick(self.symbol)

            if not tick:
                return False

            # 深度数据验证

            if not self._validate_tick_quality(tick):
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

            logger.error(f"处理Tick数据异常: {str(e)}")

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

            # 缓存计算结果

            self.indicators_cache = indicators.copy()

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

        """分析EMA排列"""

        periods = sorted(ema_series.keys())

        if len(periods) < 3:
            return 0.0

        # 检查多头排列

        is_bullish = all(ema_series[periods[i]] < ema_series[periods[i + 1]] for i in range(len(periods) - 1))

        is_bearish = all(ema_series[periods[i]] > ema_series[periods[i + 1]] for i in range(len(periods) - 1))

        if is_bullish and current_price > ema_series[periods[-1]]:

            return 0.9  # 强多头

        elif is_bearish and current_price < ema_series[periods[-1]]:

            return -0.9  # 强空头

        elif is_bullish:

            return 0.6  # 弱多头

        elif is_bearish:

            return -0.6  # 弱空头

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

            state_probabilities = {

                'TRENDING': self._calculate_trending_probability(indicators),

                'RANGING': self._calculate_ranging_probability(indicators),

                'VOLATILE': self._calculate_volatile_probability(indicators),

                'UNCERTAIN': 0.1  # 基础不确定性

            }

            # 选择最可能的状态

            max_state = max(state_probabilities, key=state_probabilities.get)

            max_prob = state_probabilities[max_state]

            # 状态转换逻辑

            if max_prob > 0.6 and max_state != self.current_state:

                old_state = self.current_state

                self.current_state = max_state

                self.state_confidence = max_prob

                self.last_state_change = time.time()

                self.state_duration = 0

                # 记录状态变更

                state_record = {

                    'timestamp': time.time(),

                    'from_state': old_state,

                    'to_state': max_state,

                    'confidence': max_prob,

                    'duration': self.state_duration

                }

                self.state_history.append(state_record)

                logger.info(f"🔄 市场状态变更: {old_state} -> {max_state} (置信度: {max_prob:.2f})")

            else:

                self.state_duration = time.time() - self.last_state_change

            return self.current_state, self.state_confidence

        except Exception as e:

            logger.error(f"分析市场状态异常: {str(e)}")

            return "UNCERTAIN", 0.0

    def _calculate_trending_probability(self, indicators: Dict) -> float:

        """计算趋势市概率"""

        probability = 0.0

        weight_sum = 0.0

        try:

            # ADX趋势强度

            adx = indicators.get('ADX', 0)

            if adx > ProfessionalComplexConfig.MARKET_STATE_PARAMS['TRENDING']['ADX_THRESHOLD']:
                adx_score = min(1.0, adx / 50.0)

                probability += adx_score * 0.25

                weight_sum += 0.25

            # EMA排列趋势

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

        """计算震荡市概率"""

        probability = 0.0

        weight_sum = 0.0

        try:

            # 低波动率

            atr_percent = indicators.get('ATR_PERCENT', 0)

            if atr_percent < ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['ATR_RATIO_MAX']:
                low_vol_score = 1.0 - (

                        atr_percent / ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['ATR_RATIO_MAX'])

                probability += low_vol_score * 0.30

                weight_sum += 0.30

            # 布林带收缩

            bb_width = indicators.get('BB_WIDTH_RATIO', 0)

            if bb_width < ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['BB_WIDTH_RATIO']:
                bb_score = 1.0 - (bb_width / ProfessionalComplexConfig.MARKET_STATE_PARAMS['RANGING']['BB_WIDTH_RATIO'])

                probability += bb_score * 0.25

                weight_sum += 0.25

            # ADX低值

            adx = indicators.get('ADX', 0)

            if adx < 20:
                adx_score = 1.0 - (adx / 20.0)

                probability += adx_score * 0.20

                weight_sum += 0.20

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

            return probability / weight_sum if weight_sum > 0 else 0.0

        except Exception as e:

            logger.warning(f"计算震荡概率异常: {str(e)}")

            return 0.0

    def _calculate_volatile_probability(self, indicators: Dict) -> float:

        """计算高波动市概率"""

        probability = 0.0

        weight_sum = 0.0

        try:

            # 高波动率

            atr_percent = indicators.get('ATR_PERCENT', 0)

            if atr_percent > ProfessionalComplexConfig.MARKET_STATE_PARAMS['VOLATILE']['ATR_RATIO_MIN']:
                high_vol_score = min(1.0, atr_percent / 0.001)

                probability += high_vol_score * 0.35

                weight_sum += 0.35

            # 布林带扩张

            bb_width = indicators.get('BB_WIDTH_RATIO', 0)

            if bb_width > 0.003:
                width_score = min(1.0, bb_width / 0.005)

                probability += width_score * 0.25

                weight_sum += 0.25

            # 价格大幅变动

            prices = list(self.data_engine.price_buffer)

            if len(prices) >= 10:

                max_change = max(

                    abs((prices[i] - prices[i - 1]) / prices[i - 1]) for i in range(1, min(10, len(prices))))

                if max_change > ProfessionalComplexConfig.MARKET_STATE_PARAMS['VOLATILE']['PRICE_SPIKE_FREQUENCY']:
                    change_score = min(1.0, max_change / 0.005)

                    probability += change_score * 0.25

                    weight_sum += 0.25

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

            return probability / weight_sum if weight_sum > 0 else 0.0

        except Exception as e:

            logger.warning(f"计算波动概率异常: {str(e)}")

            return 0.0


class ProfessionalSignalGenerator:
    """专业信号生成器 - 基于市场状态和多重指标"""

    def __init__(self, data_engine: ProfessionalTickDataEngine, market_analyzer: AdvancedMarketStateAnalyzer):

        self.data_engine = data_engine

        self.market_analyzer = market_analyzer

        self.last_signal_time = 0

        self.signal_history = deque(maxlen=100)

        self.confirmation_count = 0

    def generate_trading_signal(self) -> Optional[Dict[str, Any]]:

        """生成交易信号"""

        if not self.data_engine.initialized:
            return None

        try:

            # 检查信号间隔

            current_time = time.time()

            if current_time - self.last_signal_time < ProfessionalComplexConfig.SIGNAL_GENERATION['FILTERS'][
                'MIN_TICKS_BETWEEN_SIGNALS']:
                return None

            # 获取市场状态

            market_state, state_confidence = self.market_analyzer.analyze_complex_market_state()

            if state_confidence < 0.5:
                return None

            # 获取技术指标

            indicators = self.data_engine.calculate_complex_indicators()

            if not indicators:
                return None

            # 获取当前价格和点差

            current_tick = self.data_engine.tick_buffer[-1] if self.data_engine.tick_buffer else None

            if not current_tick:
                return None

            current_price = indicators.get('CURRENT_PRICE', current_tick['mid_price'])

            spread = current_tick['spread']

            # 移除点差过滤，点差信息仅用于记录

            # 根据市场状态生成信号

            signal = None

            if market_state == 'TRENDING':

                signal = self._generate_trending_signal(indicators, current_price, spread)

            elif market_state == 'RANGING':

                signal = self._generate_ranging_signal(indicators, current_price, spread)

            elif market_state == 'VOLATILE':

                signal = self._generate_volatile_signal(indicators, current_price, spread)

            if signal and signal['strength'] >= ProfessionalComplexConfig.SIGNAL_GENERATION['MIN_STRENGTH']:
                signal['market_state'] = market_state

                signal['state_confidence'] = state_confidence

                signal['timestamp'] = current_time

                self.last_signal_time = current_time

                self.signal_history.append(signal)

                logger.info(
                    f"📈 生成信号: {signal['direction']} 强度: {signal['strength']:.2f} 价格: {current_price:.2f}")

                return signal

            return None

        except Exception as e:

            logger.error(f"生成信号异常: {str(e)}")

            return None

    def _generate_trending_signal(self, indicators: Dict, current_price: float, spread: float) -> Optional[Dict]:

        """生成趋势市信号"""

        weights = ProfessionalComplexConfig.SIGNAL_GENERATION['WEIGHT_SYSTEM']['TRENDING']

        signal_score = 0.0

        direction = 0  # 1=买入, -1=卖出

        # 趋势指标分析

        ema_alignment = indicators.get('EMA_ALIGNMENT', 0)

        macd_trend = indicators.get('MACD_TREND', 0)

        adx = indicators.get('ADX', 0)

        if ema_alignment > 0.5 and macd_trend > 0.3 and adx > 20:

            signal_score += weights['TREND_INDICATORS']

            direction = 1

        elif ema_alignment < -0.5 and macd_trend < -0.3 and adx > 20:

            signal_score += weights['TREND_INDICATORS']

            direction = -1

        # 动量指标

        rsi_14 = indicators.get('RSI_14', 50)

        stoch_k = indicators.get('STOCH_K', 50)

        if direction == 1:

            if rsi_14 < 70 and stoch_k < 80:
                signal_score += weights['MOMENTUM_INDICATORS'] * 0.5

        elif direction == -1:

            if rsi_14 > 30 and stoch_k > 20:
                signal_score += weights['MOMENTUM_INDICATORS'] * 0.5

        # 波动率确认

        atr_percent = indicators.get('ATR_PERCENT', 0)

        if 0.0001 < atr_percent < 0.001:
            signal_score += weights['VOLATILITY_INDICATORS']

        if signal_score > 0 and direction != 0:
            return {

                'direction': 'BUY' if direction == 1 else 'SELL',

                'strength': min(1.0, signal_score),

                'entry_price': current_price,

                'spread': spread

            }

        return None

    def _generate_ranging_signal(self, indicators: Dict, current_price: float, spread: float) -> Optional[Dict]:

        """生成震荡市信号"""

        weights = ProfessionalComplexConfig.SIGNAL_GENERATION['WEIGHT_SYSTEM']['RANGING']

        signal_score = 0.0

        direction = 0

        # 震荡指标分析

        rsi_14 = indicators.get('RSI_14', 50)

        stoch_k = indicators.get('STOCH_K', 50)

        stoch_d = indicators.get('STOCH_D', 50)

        williams = indicators.get('WILLIAMSR', -50)

        # 超卖买入

        if rsi_14 < 30 and stoch_k < 20 and williams < -80:

            signal_score += weights['OSCILLATORS']

            direction = 1

        # 超买卖出

        elif rsi_14 > 70 and stoch_k > 80 and williams > -20:

            signal_score += weights['OSCILLATORS']

            direction = -1

        # 布林带位置

        bb_position = indicators.get('BB_POSITION', 0.5)

        if direction == 1 and bb_position < 0.2:

            signal_score += weights['SUPPORT_RESISTANCE'] * 0.5

        elif direction == -1 and bb_position > 0.8:

            signal_score += weights['SUPPORT_RESISTANCE'] * 0.5

        if signal_score > 0 and direction != 0:
            return {

                'direction': 'BUY' if direction == 1 else 'SELL',

                'strength': min(1.0, signal_score),

                'entry_price': current_price,

                'spread': spread

            }

        return None

    def _generate_volatile_signal(self, indicators: Dict, current_price: float, spread: float) -> Optional[Dict]:

        """生成高波动市信号"""

        weights = ProfessionalComplexConfig.SIGNAL_GENERATION['WEIGHT_SYSTEM']['VOLATILE']

        signal_score = 0.0

        direction = 0

        # 突破信号

        bb_upper = indicators.get('BB_UPPER_2.0', current_price)

        bb_lower = indicators.get('BB_LOWER_2.0', current_price)

        if current_price > bb_upper * 0.999:

            signal_score += weights['BREAKOUT_SIGNALS']

            direction = 1

        elif current_price < bb_lower * 1.001:

            signal_score += weights['BREAKOUT_SIGNALS']

            direction = -1

        # 价格行为确认

        prices = list(self.data_engine.price_buffer)

        if len(prices) >= 5:

            recent_momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0

            if direction == 1 and recent_momentum > 0.0005:

                signal_score += weights['PRICE_ACTION']

            elif direction == -1 and recent_momentum < -0.0005:

                signal_score += weights['PRICE_ACTION']

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

    def calculate_position_size(self, signal: Dict, entry_price: float) -> float:

        """计算仓位大小"""

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

            # 仓位计算: 风险金额 / (止损距离 * 每点价值)

            lot_size = risk_amount / (stop_loss_distance * tick_value)

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

            # 限制在合理范围

            lot_size = max(ProfessionalComplexConfig.MIN_LOT,

                           min(ProfessionalComplexConfig.MAX_LOT, lot_size))

            # 四舍五入到步长

            lot_size = round(lot_size / ProfessionalComplexConfig.LOT_STEP) * ProfessionalComplexConfig.LOT_STEP

            return lot_size

        except Exception as e:

            logger.error(f"计算仓位大小异常: {str(e)}")

            return ProfessionalComplexConfig.MIN_LOT

    def calculate_stop_loss_distance(self, signal: Dict, entry_price: float) -> float:

        """计算止损距离"""

        try:

            indicators = self.data_engine.calculate_complex_indicators()

            atr = indicators.get('ATR', entry_price * 0.001)

            # 使用ATR倍数

            atr_multiplier = 1.5  # 默认1.5倍ATR

            stop_loss_distance = atr * atr_multiplier

            # 转换为点数

            point = self.data_engine.data_validator.symbol_info.point if self.data_engine.data_validator.symbol_info else 0.01

            stop_loss_points = stop_loss_distance / point

            return stop_loss_points

        except Exception as e:

            logger.error(f"计算止损距离异常: {str(e)}")

            return 50  # 默认50点

    def calculate_take_profit_levels(self, signal: Dict, entry_price: float, stop_loss: float) -> List[Dict]:

        """计算止盈目标"""

        try:

            risk_reward_ratio = 2.0  # 风险回报比

            base_profit = abs(entry_price - stop_loss) * risk_reward_ratio

            targets = []

            if signal['direction'] == 'BUY':

                tp1 = entry_price + base_profit * 0.5

                tp2 = entry_price + base_profit * 1.0

                tp3 = entry_price + base_profit * 1.5

            else:

                tp1 = entry_price - base_profit * 0.5

                tp2 = entry_price - base_profit * 1.0

                tp3 = entry_price - base_profit * 1.5

            targets = [

                {'price': tp1, 'close_percent': 0.25},

                {'price': tp2, 'close_percent': 0.35},

                {'price': tp3, 'close_percent': 0.40}

            ]

            return targets

        except Exception as e:

            logger.error(f"计算止盈目标异常: {str(e)}")

            return []

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

            # 清理已平仓的持仓的多目标止盈信息

            closed_tickets = set(self.open_positions.keys()) - set(new_positions.keys())

            for ticket in closed_tickets:

                if ticket in self.position_tp_targets:
                    del self.position_tp_targets[ticket]

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

    def can_open_new_position(self) -> bool:

        """检查是否可以开新仓"""

        # 检查每日交易限制

        current_date = datetime.now().date()

        if self.last_trade_date != current_date:
            self.daily_trades = 0

            self.last_trade_date = current_date

        if self.daily_trades >= ProfessionalComplexConfig.MAX_DAILY_TRADES:
            logger.warning(f"⚠️ 达到每日交易限制: {self.daily_trades}")

            return False

        # 检查并发持仓限制

        self.get_open_positions()

        if len(self.open_positions) >= ProfessionalComplexConfig.MAX_CONCURRENT_TRADES:
            logger.warning(f"⚠️ 达到最大并发持仓: {len(self.open_positions)}")

            return False

        # 检查风险限制

        if not self.risk_manager.check_risk_limits():
            return False

        return True

    def open_position(self, signal: Dict) -> Optional[int]:

        """开仓 - 使用先下单后设置止盈止损的方式"""

        if not self.can_open_new_position():
            return None

        try:

            symbol = self.data_engine.symbol

            symbol_info = self.data_engine.data_validator.symbol_info

            if not symbol_info:
                logger.error("无法获取品种信息")

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

            # 计算仓位大小

            lot_size = self.risk_manager.calculate_position_size(signal, entry_price)

            # 计算止损止盈（但先不下单时设置）

            stop_loss_distance = self.risk_manager.calculate_stop_loss_distance(signal, entry_price)

            point = symbol_info.point

            if signal['direction'] == 'BUY':

                sl_price = entry_price - stop_loss_distance * point

                tp_levels = self.risk_manager.calculate_take_profit_levels(signal, entry_price, sl_price)

                tp_price = tp_levels[0]['price'] if tp_levels else entry_price + stop_loss_distance * point * 2

            else:

                sl_price = entry_price + stop_loss_distance * point

                tp_levels = self.risk_manager.calculate_take_profit_levels(signal, entry_price, sl_price)

                tp_price = tp_levels[0]['price'] if tp_levels else entry_price - stop_loss_distance * point * 2

            # 规范化价格

            sl_price = round(sl_price / point) * point

            tp_price = round(tp_price / point) * point

            # 验证止损止盈价格是否符合品种要求
            # 获取最小止损距离（点数）
            # MT5可能使用trade_stops_level属性，如果没有则使用2个点差
            stops_level = 0
            try:
                # 尝试使用trade_stops_level属性
                if hasattr(symbol_info, 'trade_stops_level'):
                    stops_level = symbol_info.trade_stops_level
                elif hasattr(symbol_info, 'stops_level'):
                    stops_level = symbol_info.stops_level
            except:
                pass

            # 如果仍然为0，则使用当前点差的2倍（MT5黄金要求至少2个点差）
            if stops_level <= 0:
                current_spread = (symbol_info.ask - symbol_info.bid) / point  # 当前点差（点数）
                stops_level = max(2, int(current_spread * 2))  # 至少2个点差，或当前点差的2倍
                logger.debug(f"使用计算的最小止损距离: {stops_level}点（当前点差: {current_spread:.1f}点）")

            if stops_level > 0:

                # 计算止损和止盈距离入场价格的点数

                if signal['direction'] == 'BUY':

                    sl_distance_points = (entry_price - sl_price) / point

                    tp_distance_points = (tp_price - entry_price) / point

                else:

                    sl_distance_points = (sl_price - entry_price) / point

                    tp_distance_points = (entry_price - tp_price) / point

                # 确保止损和止盈距离符合最小要求

                if sl_distance_points < stops_level:

                    # 调整止损价格以满足最小距离要求

                    if signal['direction'] == 'BUY':

                        sl_price = entry_price - stops_level * point

                    else:

                        sl_price = entry_price + stops_level * point

                    sl_price = round(sl_price / point) * point

                    logger.debug(f"调整止损价格以满足最小距离要求: {stops_level}点")

                if tp_distance_points < stops_level:

                    # 调整止盈价格以满足最小距离要求

                    if signal['direction'] == 'BUY':

                        tp_price = entry_price + stops_level * point

                    else:

                        tp_price = entry_price - stops_level * point

                    tp_price = round(tp_price / point) * point

                    logger.debug(f"调整止盈价格以满足最小距离要求: {stops_level}点")

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
                logger.error(f"开仓失败: {result.retcode} - {result.comment}")

                return None

            # 获取实际成交的订单号

            order_ticket = result.order

            if not order_ticket:
                logger.error("开仓成功但未获取到订单号")

                return None

            logger.info(f"✅ 开仓成功: {signal['direction']} {lot_size}手 @ {entry_price:.2f} (订单号: {order_ticket})")

            # 保存多目标止盈信息

            if tp_levels and len(tp_levels) > 0:

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

                    normalized_price = round(tp_level['price'] / point) * point

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
                        normalized_price = round(normalized_price / point) * point
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

            # 第二步：立即设置止盈止损

            # 等待一小段时间确保订单已完全建立并转换为持仓

            time.sleep(0.2)

            # 获取持仓ticket（订单ticket和持仓ticket可能不同）

            position_ticket = None

            positions = mt5.positions_get(symbol=symbol)

            if positions:

                for pos in positions:
                    # 通过订单号或价格匹配找到对应的持仓
                    if (hasattr(pos, 'identifier') and pos.identifier == order_ticket) or \
                            (pos.type == order_type and abs(pos.price_open - entry_price) < point * 10):
                        position_ticket = pos.ticket
                        break

            if not position_ticket:
                # 如果找不到持仓，尝试使用订单号（某些情况下可能相同）

                logger.warning(f"⚠️ 未找到对应持仓，尝试使用订单号: {order_ticket}")

                position_ticket = order_ticket

            # 使用 OrderModify 设置止盈止损
            # 获取实际持仓信息，使用实际入场价格重新验证止盈止损
            positions = mt5.positions_get(symbol=symbol)
            actual_entry_price = entry_price
            if positions:
                for pos in positions:
                    if pos.ticket == position_ticket:
                        actual_entry_price = pos.price_open
                        # 使用实际入场价格重新验证和调整止盈止损
                        point = symbol_info.point
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
                            stops_level = max(2, int(current_spread * 2))

                        logger.debug(
                            f"验证止盈止损: 入场价={actual_entry_price:.2f}, 方向={signal['direction']}, 最小距离={stops_level}点")

                        # 重新验证止损
                        if sl_price > 0:
                            if signal['direction'] == 'BUY':
                                sl_distance = (actual_entry_price - sl_price) / point
                                if sl_price >= actual_entry_price or sl_distance < stops_level:
                                    old_sl = sl_price
                                    sl_price = actual_entry_price - stops_level * point
                                    sl_price = round(sl_price / point) * point
                                    logger.debug(f"调整止损: {old_sl:.2f} -> {sl_price:.2f} (距离: {stops_level}点)")
                            else:  # SELL
                                sl_distance = (sl_price - actual_entry_price) / point
                                if sl_price <= actual_entry_price or sl_distance < stops_level:
                                    old_sl = sl_price
                                    sl_price = actual_entry_price + stops_level * point
                                    sl_price = round(sl_price / point) * point
                                    logger.debug(f"调整止损: {old_sl:.2f} -> {sl_price:.2f} (距离: {stops_level}点)")

                            # 最终验证止损方向
                            if signal['direction'] == 'BUY' and sl_price >= actual_entry_price:
                                logger.warning(f"⚠️ 止损价格无效（BUY订单止损应低于入场价），跳过设置止损")
                                sl_price = 0
                            elif signal['direction'] == 'SELL' and sl_price <= actual_entry_price:
                                logger.warning(f"⚠️ 止损价格无效（SELL订单止损应高于入场价），跳过设置止损")
                                sl_price = 0

                        # 重新验证止盈
                        if tp_price > 0:
                            if signal['direction'] == 'BUY':
                                tp_distance = (tp_price - actual_entry_price) / point
                                if tp_price <= actual_entry_price or tp_distance < stops_level:
                                    old_tp = tp_price
                                    tp_price = actual_entry_price + stops_level * point
                                    tp_price = round(tp_price / point) * point
                                    logger.debug(f"调整止盈: {old_tp:.2f} -> {tp_price:.2f} (距离: {stops_level}点)")
                            else:  # SELL
                                tp_distance = (actual_entry_price - tp_price) / point
                                if tp_price >= actual_entry_price or tp_distance < stops_level:
                                    old_tp = tp_price
                                    tp_price = actual_entry_price - stops_level * point
                                    tp_price = round(tp_price / point) * point
                                    logger.debug(f"调整止盈: {old_tp:.2f} -> {tp_price:.2f} (距离: {stops_level}点)")

                            # 最终验证止盈方向
                            if signal['direction'] == 'BUY' and tp_price <= actual_entry_price:
                                logger.warning(f"⚠️ 止盈价格无效（BUY订单止盈应高于入场价），跳过设置止盈")
                                tp_price = 0
                            elif signal['direction'] == 'SELL' and tp_price >= actual_entry_price:
                                logger.warning(f"⚠️ 止盈价格无效（SELL订单止盈应低于入场价），跳过设置止盈")
                                tp_price = 0

                        logger.debug(f"最终止盈止损: SL={sl_price:.2f}, TP={tp_price:.2f}")
                        break

            # 只设置有效的止损和止盈
            if sl_price == 0 and tp_price == 0:
                logger.warning(f"⚠️ 止损和止盈都无效，跳过设置")
                self.daily_trades += 1
                return order_ticket

            modify_request = {

                "action": mt5.TRADE_ACTION_SLTP,

                "symbol": symbol,

                "position": position_ticket,

            }

            if sl_price > 0:
                modify_request["sl"] = sl_price

            if tp_price > 0:
                modify_request["tp"] = tp_price

            logger.debug(f"发送止盈止损设置请求: {modify_request}")
            modify_result = mt5.order_send(modify_request)

            if modify_result is None:

                error_code = mt5.last_error()

                logger.warning(f"⚠️ 止盈止损设置失败: order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")

                # 如果失败，再等待一下并重试

                time.sleep(0.2)

                positions = mt5.positions_get(symbol=symbol)

                if positions:

                    for pos in positions:

                        if pos.type == order_type and abs(pos.price_open - entry_price) < point * 10:

                            modify_request["position"] = pos.ticket

                            modify_result = mt5.order_send(modify_request)

                            if modify_result is None:

                                error_code = mt5.last_error()

                                logger.warning(
                                    f"⚠️ 重试后仍失败: order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")

                            elif modify_result.retcode == mt5.TRADE_RETCODE_DONE:

                                logger.info(f"✅ 重试后止盈止损设置成功: SL:{sl_price:.2f} TP:{tp_price:.2f}")

                            else:

                                logger.warning(f"⚠️ 重试后仍失败: {modify_result.retcode} - {modify_result.comment}")

                            break

            elif modify_result.retcode == mt5.TRADE_RETCODE_DONE:

                logger.info(f"✅ 止盈止损设置成功: SL:{sl_price:.2f} TP:{tp_price:.2f}")

            else:

                logger.warning(f"⚠️ 止盈止损设置失败: {modify_result.retcode} - {modify_result.comment}")

                # 如果失败，再等待一下并重试

                time.sleep(0.2)

                positions = mt5.positions_get(symbol=symbol)

                if positions:

                    for pos in positions:

                        if pos.type == order_type and abs(pos.price_open - entry_price) < point * 10:

                            modify_request["position"] = pos.ticket

                            modify_result = mt5.order_send(modify_request)

                            if modify_result is None:

                                error_code = mt5.last_error()

                                logger.warning(
                                    f"⚠️ 重试后仍失败: order_send返回None，错误代码: {error_code[0]} - {error_code[1]}")

                            elif modify_result.retcode == mt5.TRADE_RETCODE_DONE:

                                logger.info(f"✅ 重试后止盈止损设置成功: SL:{sl_price:.2f} TP:{tp_price:.2f}")

                            else:

                                logger.warning(f"⚠️ 重试后仍失败: {modify_result.retcode} - {modify_result.comment}")

                            break

            self.daily_trades += 1

            return order_ticket

        except Exception as e:

            logger.error(f"开仓异常: {str(e)}")

            traceback.print_exc()

            return None

    def update_positions(self):

        """更新持仓状态（跟踪止损、多目标止盈等）"""

        try:

            positions = self.get_open_positions()

            indicators = self.data_engine.calculate_complex_indicators()

            current_price = indicators.get('CURRENT_PRICE', 0)

            if not current_price:
                return

            for ticket, pos in positions.items():

                # 检查多目标止盈

                self._check_multi_target_take_profit(ticket, pos, current_price)

                # 检查是否需要移动止损

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

            current_sl = position.get('sl', 0)

            if position['type'] == 'BUY':

                profit_percent = (current_price - entry_price) / entry_price

                if profit_percent >= activation_percent:

                    new_sl = current_price - step_size * entry_price

                    if new_sl > current_sl:
                        self._modify_stop_loss(ticket, new_sl)

            else:

                profit_percent = (entry_price - current_price) / entry_price

                if profit_percent >= activation_percent:

                    new_sl = current_price + step_size * entry_price

                    if new_sl < current_sl or current_sl == 0:
                        self._modify_stop_loss(ticket, new_sl)

        except Exception as e:

            logger.debug(f"更新跟踪止损异常: {str(e)}")

    def _modify_stop_loss(self, ticket: int, new_sl: float):

        """修改止损"""

        try:

            request = {

                "action": mt5.TRADE_ACTION_SLTP,

                "position": ticket,

                "sl": new_sl,

            }

            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.debug(f"✅ 止损已更新: {ticket} -> {new_sl:.2f}")

        except Exception as e:

            logger.debug(f"修改止损异常: {str(e)}")

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

                            logger.info(f"🎯 达到止盈目标TP{i + 1} ({tp_price:.2f})，部分平仓: {close_volume}手")

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

        except Exception as e:

            logger.error(f"检查多目标止盈异常: {str(e)}")

            traceback.print_exc()

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
                stops_level = max(2, int(current_spread * 2))

            # 验证止盈价格
            entry_price = position.price_open
            position_type = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

            # 计算止盈距离
            if position_type == 'BUY':
                tp_distance = (new_tp - entry_price) / point
                # BUY订单：止盈应高于入场价，且距离至少为stops_level
                if new_tp <= entry_price:
                    logger.warning(f"⚠️ 止盈价格无效（BUY订单止盈应高于入场价 {entry_price:.2f}），跳过更新")
                    return
                if tp_distance < stops_level:
                    # 调整止盈价格
                    new_tp = entry_price + stops_level * point
                    new_tp = round(new_tp / point) * point
                    logger.debug(f"调整止盈价格以满足最小距离要求: {stops_level}点")
            else:  # SELL
                tp_distance = (entry_price - new_tp) / point
                # SELL订单：止盈应低于入场价，且距离至少为stops_level
                if new_tp >= entry_price:
                    logger.warning(f"⚠️ 止盈价格无效（SELL订单止盈应低于入场价 {entry_price:.2f}），跳过更新")
                    return
                if tp_distance < stops_level:
                    # 调整止盈价格
                    new_tp = entry_price - stops_level * point
                    new_tp = round(new_tp / point) * point
                    logger.debug(f"调整止盈价格以满足最小距离要求: {stops_level}点")

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

        self.running = False

        self.processing_thread = None

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

            analysis_interval = 1.0  # 每秒分析一次

            while self.running:

                try:

                    current_time = time.time()

                    # 处理Tick数据

                    self.data_engine.process_tick_data()

                    # 定期分析（降低频率）

                    if current_time - last_analysis_time >= analysis_interval:

                        # 更新账户信息

                        self.risk_manager.update_account_info()

                        # 更新持仓状态

                        self.position_manager.update_positions()

                        # 生成交易信号

                        signal = self.signal_generator.generate_trading_signal()

                        if signal:
                            # 尝试开仓

                            self.position_manager.open_position(signal)

                        last_analysis_time = current_time

                    time.sleep(ProfessionalComplexConfig.PROCESSING_INTERVAL)

                except KeyboardInterrupt:

                    logger.info("⚠️ 收到中断信号，停止策略...")

                    self.running = False

                    break

                except Exception as e:

                    logger.error(f"策略循环异常: {str(e)}")

                    traceback.print_exc()

                    time.sleep(1)

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
