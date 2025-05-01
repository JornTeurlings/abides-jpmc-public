import importlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple
from abc import ABC

import gymnasium as gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator

from .markets_environment import AbidesGymMarketsEnv
from abides_markets.orders import LimitOrder


class SubGymMarketsExecutionEnvThesis_v0(AbidesGymMarketsEnv):
    """
    Execution V0 environment. It defines one of the ABIDES-Gym-markets environment.
    This environment presents an example of the algorithmic order execution problem.
    The agent has either an initial inventory of the stocks it tries to trade out of or no initial inventory and
    tries to acquire a target number of shares. The goal is to realize this task while minimizing transaction cost from spreads
     and market impact. It does so by splitting the parent order into several smaller child orders.

    Arguments:
        - background_config: the handcrafted agents configuration used for the environment
        - mkt_close: time the market day ends
        - timestep_duration: how long between 2 wakes up of the gym experimental agent
        - starting_cash: cash of the agents at the beginning of the simulation
        - order_fixed_size: size of the order placed by the experimental gym agent
        - state_history_length: length of the raw state buffer
        - market_data_buffer_length: length of the market data buffer
        - first_interval: how long the simulation is run before the first wake up of the gym experimental agent
        - parent_order_size: Total size the agent has to execute (either buy or sell).
        - execution_window: Time length the agent is given to proceed with 𝑝𝑎𝑟𝑒𝑛𝑡𝑂𝑟𝑑𝑒𝑟𝑆𝑖𝑧𝑒execution.
        - direction: direction of the 𝑝𝑎𝑟𝑒𝑛𝑡𝑂𝑟𝑑𝑒𝑟 (buy or sell)
        - not_enough_reward_update: it is a constant penalty per non-executed share at the end of the𝑡𝑖𝑚𝑒𝑊𝑖𝑛𝑑𝑜𝑤
        - just_quantity_reward_update: update reward if all order is completed
        - reward_mode: can use a dense of sparse reward formulation
        - done_ratio: ratio (mark2market_t/starting_cash) that defines when an episode is done (if agent has lost too much mark to market value)
        - debug_mode: arguments to change the info dictionary (lighter version if performance is an issue)
        - background_config_extra_kvargs: dictionary of extra key value  arguments passed to the background config builder function

    Daily Investor V0:
        - Action Space:
            - LMT variable_size variable_price
            - Hold
        - State Space:
            - holdings_pct
            - time_pct
            - diff_pct
            - imbalance_all
            - imbalance_5
            - price_impact
            - spread
            - direction
            - returns
    """

    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )

    @dataclass
    class CustomMetricsTracker(ABC):
        """
        Data Class used to track custom metrics that are output to rllib
        """

        executed_quantity: int = 0  # at the end of the episode
        remaining_quantity: int = 0  # at the end of the episode

        action_counter: Dict[str, int] = field(default_factory=dict)

        holdings_pct: float = 0
        time_pct: float = 0
        diff_pct: float = 0
        imbalance_all: float = 0
        short_term_vol: float = 0
        price_impact: int = 0
        spread: int = 0
        top_of_the_book_liquidity: float = 0
        num_max_steps_per_episode: float = 0
        depth: int = 0
        mtm_pnl: float = 0
        cash: float = 0
        holdings: float = 0
        ml_ofi: list[float] = field(default_factory=list)

        dp_t: float = 0
        ip_t: float = 0
        tp_t: float = 0
        fr_t: float = 0
        cp_t: float = 0
        tip_t: float = 0
        bid_penalty: float = 0
        ask_penalty: float = 0
        reward: float = 0
        reward_components: Dict[str, float] = field(default_factory=dict)

    def __init__(
            self,
            background_config: Any = "rmsc04",
            mkt_close: str = "16:00:00",
            timestep_duration: str = "60s",
            starting_cash: int = 10_000_000,
            order_fixed_size: int = 20,
            state_history_length: int = 10,
            market_data_buffer_length: int = 5,
            first_interval: str = "00:00:30",
            parent_order_size: int = 1200,
            scale_price: int = 100000,
            execution_window: str = "00:10:00",
            direction: str = "BUY",
            not_enough_reward_update: int = -100,
            too_much_reward_update: int = -80,
            just_quantity_reward_update: int = 0,
            debug_mode: bool = False,
            tuning_params: Dict[str, any] = {},
            domain_randomization_envs: List[str] = [],
            background_config_extra_kvargs: Dict[str, Any] = {},
            saved_models_location: str | None = None,
            uses_beta: bool = False
    ) -> None:
        if background_config == 'random':
            self.background_config = [
                importlib.import_module("abides_markets.configs.{}".format(config), package=None).build_config
                for config in domain_randomization_envs]
        else:
            self.background_config: Any = importlib.import_module(
                "abides_markets.configs.{}".format(background_config), package=None
            ).build_config
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.starting_cash: int = starting_cash
        self.scale_price: int = scale_price
        self.order_fixed_size: int = order_fixed_size
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.parent_order_size: int = parent_order_size
        self.execution_window: int = str_to_ns(execution_window)
        self.direction: str = direction
        self.debug_mode: bool = debug_mode
        self.uses_beta = uses_beta

        self.too_much_reward_update: int = too_much_reward_update
        self.not_enough_reward_update: int = not_enough_reward_update
        self.just_quantity_reward_update: int = just_quantity_reward_update

        self.entry_price: int = 1
        self.far_touch: int = 1
        self.near_touch: int = 1
        self.step_index: int = 0

        self.custom_metrics_tracker = (
            self.CustomMetricsTracker()
        )  # init the custom metric tracker

        ##################
        # CHECK PROPERTIES
        assert background_config in [
            "rmsc03",
            "rmsc04",
            "smc_01",
            "test",
            "momentum",
            "high_liq",
            "high_vol_shock",
            "low_info",
            "exec_pressure",
            "low_liq",
            "random",
            "single_agent_env"
        ], "No correct config selected as config"

        assert (self.first_interval <= str_to_ns("16:00:00")) & (
                self.first_interval >= str_to_ns("00:00:00")
        ), "Select authorized FIRST_INTERVAL delay"

        assert (self.mkt_close <= str_to_ns("16:00:00")) & (
                self.mkt_close >= str_to_ns("09:30:00")
        ), "Select authorized market hours"

        assert (self.timestep_duration <= str_to_ns("06:30:00")) & (
                self.timestep_duration >= str_to_ns("00:00:00")
        ), "Select authorized timestep_duration"

        assert (type(self.starting_cash) == int) & (
                self.starting_cash >= 0
        ), "Select positive integer value for starting_cash"

        assert (type(self.order_fixed_size) == int) & (
                self.order_fixed_size >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.state_history_length) == int) & (
                self.state_history_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.market_data_buffer_length) == int) & (
                self.market_data_buffer_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert self.debug_mode in [
            True,
            False,
        ], "debug_mode needs to be True or False"

        assert self.direction in [
            "BUY",
            "SELL",
        ], "direction needs to be BUY or SELL"

        assert (type(self.parent_order_size) == int) & (
                self.order_fixed_size >= 0
        ), "Select positive integer value for parent_order_size"

        assert (self.execution_window <= str_to_ns("06:30:00")) & (
                self.execution_window >= str_to_ns("00:00:00")
        ), "Select authorized execution_window"

        assert (
                type(self.too_much_reward_update) == int
        ), "Select integer value for too_much_reward_update"

        assert (
                type(self.not_enough_reward_update) == int
        ), "Select integer value for not_enough_reward_update"
        assert (
                type(self.just_quantity_reward_update) == int
        ), "Select integer value for just_quantity_reward_update"

        background_config_args = {"end_time": self.mkt_close}
        background_config_args.update(background_config_extra_kvargs)
        super().__init__(
            background_config_pair=(
                self.background_config,
                background_config_args,
            ),
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=self.timestep_duration
            ),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval,
            saved_models_location=saved_models_location
        )

        # Action Space

        # Bid above midprice | bid size | Ask above midprice | ask size
        self.multiple_action_heads = True
        self.num_actions: int = 2
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            shape=(self.num_actions,),
            dtype=np.float32
        )

        # instantiate the action counter
        for i in range(self.num_actions):
            self.custom_metrics_tracker.action_counter[f"action_{i}"] = 0

        num_ns_episode = self.first_interval + self.execution_window
        step_length = self.timestep_duration
        num_max_steps_per_episode = num_ns_episode / step_length
        self.custom_metrics_tracker.num_max_steps_per_episode = (
            num_max_steps_per_episode
        )

        # Dynamically determine the number of MLOFI levels to be included
        self.mlofi_depth = tuning_params.get('mlofi_depth', 1)  # Default to 1 if not specified
        self.ofi_lag = tuning_params.get('ofi_lag', 1)

        assert (self.ofi_lag < self.state_history_length), 'OFI Lag must be smaller than data to calculate it'

        # Update state feature count to reflect the new state variables
        self.num_state_features: int = 9 + self.state_history_length - 1 + self.ofi_lag + self.mlofi_depth  # 9 fixed features + OFI + returns + MLOFI levels

        # Define upper bounds for state values
        self.state_highs: np.ndarray = np.array(
            [
                2,  # holdings_pct
                np.finfo(np.float32).max,  # scaled_mid_price
                1,  # time_pct
                3,  # diff_pct
                1,  # imbalance_all
                np.finfo(np.float32).max,  # spread
                1,  # short_term_vol
                1,  # top_of_book_liquidity
                10,  # depth (set an upper bound for depth)
            ]
            + [1] * self.mlofi_depth  # Dynamic MLOFI levels
            + self.ofi_lag * [1]  # Lagged time OFI
            + (self.state_history_length - 1) * [np.finfo(np.float32).max],  # Returns
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        # Define lower bounds for state values
        self.state_lows: np.ndarray = np.array(
            [
                -2,  # holdings_pct
                0,  # scaled_mid_price
                0,  # time_pct
                -2,  # diff_pct
                -1,  # imbalance_all
                np.finfo(np.float32).min,  # spread
                0,  # short_term_vol
                0,  # top_of_book_liquidity
                0,  # depth (depth cannot be negative)
            ]
            + [-1] * self.mlofi_depth  # Dynamic MLOFI levels
            + self.ofi_lag * [-1]  # Lagged time OFI
            + (self.state_history_length - 1) * [np.finfo(np.float32).min],  # Returns
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.observation_space: gym.Space = gym.spaces.Box(
            self.state_lows,
            self.state_highs,
            shape=(self.num_state_features, 1),
            dtype=np.float32,
        )
        # initialize previous_marked_to_market to starting_cash (No holding at the beginning of the episode)
        self.previous_marked_to_market: int = self.starting_cash

        # Penalty for the inventory
        self.out_of_spread_error: float = tuning_params.get('out_of_spread_error', 0.5)
        self.fill_ratio_bonus: float = tuning_params.get("fill_ratio_bonus", 100)
        self.tpt_weight: float = tuning_params.get('tpt_weight', 0.001)
        self.dpt_eta: float = tuning_params.get('dpt_eta', 0.2)
        self.dpt_weight: float = tuning_params.get('dpt_weight', 0.02)
        self.inventory_penalty: float = tuning_params.get("inventory_penalty", 1)
        self.terminal_inventory_penalty: float = tuning_params.get("terminal_inventory_penalty", 1)
        self.orders_submitted: int = 0
        self.previous_asks: list | None = None
        self.previous_bids: list | None = None
        self.previous_depth: int = 0
        self.reservation_quote = tuning_params.get('reservation_quote', 0.01)
        self.max_spread = tuning_params.get('max_spread', 0.001)
        self.direct_action = tuning_params.get('direction_action', False)

        self.kappa = tuning_params.get('kappa', 0.05)
        self.lambda_r = tuning_params.get('lambda_r', 1e-3)

        self.scale_rew = tuning_params.get('scale_rew', False)
        self.dpt_importance = tuning_params.get('dpt_importance', 1)
        self.tpt_importance = tuning_params.get('tpt_importance', 1)
        self.cpt_importance = tuning_params.get('cpt_importance', 1)
        self.ipt_importance = tuning_params.get('ipt_importance', 1)
        self.frt_importance = tuning_params.get('frt_importance', 1)

        # Some default values for initialization
        self.last_spread = 0
        self.last_bid_action = 0
        self.last_ask_action = 0
        self.last_action_1 = 0
        self.last_action_2 = 0

        environment_config = {
            'parent_order_size': self.parent_order_size,
            'execution_window': self.execution_window,
            'scale_price': self.scale_price,
            'state_history_length': self.state_history_length,
            'mlofi_depth': self.mlofi_depth,
            'ofi_lag': self.ofi_lag,
            'num_state_features': self.num_state_features,
            'reservation_quote': self.reservation_quote,
            'max_spread': self.max_spread,
            'direct_action': self.direct_action,
            'order_fixed_size': self.order_fixed_size
        }

        # This is needed to make self-play work
        self.set_environment_configuration(environment_config)

        self.parent_order_size = self.environment_configuration['parent_order_size']
        self.execution_window = str_to_ns(self.environment_configuration['execution_window'])
        self.scale_price = self.environment_configuration['scale_price']
        self.state_history_length = self.environment_configuration['state_history_length']
        self.mlofi_depth = self.environment_configuration['mlofi_depth']
        self.ofi_lag = self.environment_configuration['ofi_lag']
        self.num_state_features = self.environment_configuration['num_state_features']
        self.reservation_quote = self.environment_configuration['reservation_quote']
        self.max_spread = self.environment_configuration['max_spread']
        self.direct_action = self.environment_configuration.get('direct_action', False)
        self.order_fixed_size = self.environment_configuration.get('order_fixed_size', 100)

    def compute_bid_ask_reservation(self, spread_val, res_val, extra_info=False) -> tuple[
        int, int, float | None]:
        """
        Function for calculating the bid-ask spread based on the input
        Args:
            spread_val: x in (-1, 1) to indicate how wide the spread will be
            res_val: y in (-1, 1) to indicate how far off the mid-price we will quote our mid-price

        Returns: The bid-price and ask-price that will be quoted

        """
        mid_price = self.last_mid_price  # ~100,000

        if not self.uses_beta:
            # This means it uses TanH so it has to be shifted up by 1 and divided by 2
            spread_val = (spread_val + 1) / 2  # We need to stay in the range of (0, 1)
        else:
            res_val = 2 * res_val - 1  # We need to go to (-1, 1)

        # Reservation Price
        reservation_price = mid_price - self.max_spread * res_val

        # Spread (split in half around the reservation_price)
        # We need to ensure at least a width of 1
        spread_val = min(max(spread_val, 0.1), 1)
        half_spread = (spread_val * self.max_spread) / 2.0

        bid_price = round(reservation_price - half_spread)
        ask_price = round(reservation_price + half_spread)

        return bid_price, ask_price, reservation_price if extra_info else None

    def compute_bid_ask_direct(self, bid_val, ask_val) -> tuple[int, int, float | None]:
        """
        Function for calculating the bid-ask spread based on the input
        Args:
            bid_val: x in (0, 1) to indicate how much extra from the mid_price bid will be
            ask_val: y in (0, 1) to indicate how much extra from the mid_price ask will be

        Returns: The bid-price and ask-price that will be quoted

        """
        if not self.uses_beta:
            bid_val = (bid_val + 1) / 2
            ask_val = (ask_val + 1) / 2

        mid_price = self.last_mid_price  # ~100,000

        bid_price = round(mid_price * (1 - bid_val))
        ask_price = round(mid_price * (1 + ask_val))

        return bid_price, ask_price, None

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(self, action: list):
        """
        action is a 2D float array:
            [spread_val, reservation_val]

        """

        action_1, action_2 = action[:2]

        self.last_action_1 = action_1
        self.last_action_2 = action_2

        if self.direct_action:
            bid_price, ask_price, _ = self.compute_bid_ask_direct(action_1, action_2)
        else:
            bid_price, ask_price, _ = self.compute_bid_ask_reservation(action_1, action_2)

        # Build instructions
        instructions = [{"type": "CCL_ALL"}]
        instructions.append({
            "type": "LMT",
            "direction": "BUY",
            "size": self.order_fixed_size,
            "limit_price": bid_price
        })
        instructions.append({
            "type": "LMT",
            "direction": "SELL",
            "size": self.order_fixed_size,
            "limit_price": ask_price
        })

        # Set for computing reward errors in later stage
        self.last_bid_action = bid_price
        self.last_ask_action = ask_price
        self.orders_submitted += 2
        return instructions

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        Method that transforms a raw state into a dimensionless,
        well-scaled state representation for both PPO and SAC.

        Returns:
            state (np.ndarray): shape (self.num_state_features, 1)
        """
        # ---------------------------
        # 0) Preliminary
        # ---------------------------
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        bid_volume = raw_state["parsed_volume_data"]["bid_volume"][-1]
        ask_volume = raw_state["parsed_volume_data"]["ask_volume"][-1]

        # 1) Holdings (scaled)
        holdings = raw_state["internal_data"]["holdings"]
        holdings_pct = holdings[-1] / (self.parent_order_size)  # dimensionless in [-1,1]

        # 2) Timing
        mkt_open = raw_state["internal_data"]["mkt_open"][-1]
        current_time = raw_state["internal_data"]["current_time"][-1]
        time_from_parent_arrival = current_time - mkt_open - self.first_interval
        assert (current_time >= mkt_open + self.first_interval), (
            "Agent has woken up earlier than its first interval"
        )
        time_limit = self.execution_window
        time_pct = time_from_parent_arrival / time_limit  # dimensionless in [0,1]

        # 3) Compare progress vs holdings
        diff_pct = holdings_pct - time_pct  # in [-1, 1]

        # 4) Order book imbalance
        imbalances_all = [
            markets_agent_utils.get_imbalance(b, a, depth=None)
            for (b, a) in zip(bids, asks)
        ]
        imbalance_all = imbalances_all[-1]  # typically in [-1,1]

        # 5) Mid Prices + Price Impact
        mid_prices = [
            markets_agent_utils.get_mid_price(bid, ask, lt)
            for (bid, ask, lt) in zip(bids, asks, last_transactions)
        ]
        mid_price = mid_prices[-1]  # current mid

        # 6) Best bids / asks
        best_bids = [
            b[0][0] if len(b) > 0 else mp
            for (b, mp) in zip(bids, mid_prices)
        ]
        best_asks = [
            a[0][0] if len(a) > 0 else mp
            for (a, mp) in zip(asks, mid_prices)
        ]

        self.last_mid_price = (best_asks[-1] + best_bids[-1]) / 2
        # Scale mid_price to ~ 1 range if needed
        scaled_mid_price = self.last_mid_price / self.scale_price  # dimensionless around 1

        # Spread as fraction of mid
        spreads = np.array(best_asks) - np.array(best_bids)
        spread = spreads[-1]  # dimensionless relative to mid_price < 1/2

        # 7) Log returns
        # Replace raw differences with log(m_i / m_(i-1))
        log_prices = np.log(mid_prices)
        log_returns = np.diff(log_prices)

        # Pad
        padded_returns = np.zeros(self.state_history_length - 1, dtype=np.float32)
        last_k = len(log_returns)
        padded_returns[-last_k:] = log_returns if last_k > 0 else padded_returns

        # 8) Short term vol from log returns
        # e.g. std of the last 10 log returns
        if len(log_returns) >= 10:
            short_term_vol = np.std(log_returns[-10:])
        else:
            short_term_vol = 0.0
        # (Optional) Clip outliers if needed
        clip_vol = 1
        short_term_vol = float(np.clip(short_term_vol, 0.0, clip_vol))

        # 9) Liquidity & Depth
        top_bid_volume = markets_agent_utils.get_volume(bids[0], depth=1)
        top_ask_volume = markets_agent_utils.get_volume(asks[0], depth=1)
        total_lot_volume = bid_volume + ask_volume
        if total_lot_volume > 0:
            top_of_book_liquidity = min((top_bid_volume + top_ask_volume) / total_lot_volume, 1)
        else:
            top_of_book_liquidity = 0.0

        max_depth = 10
        depth = min(len(best_asks), len(best_bids)) / max_depth  # in [0,1]

        # 10) MLOFI
        if (self.previous_bids is None and self.previous_asks is None) \
                or (self.previous_bids is None) or (self.previous_asks is None):
            ml_ofi = [0.0] * self.mlofi_depth
        else:
            ml_ofi = markets_agent_utils.get_ml_ofi(
                bids[0],
                self.previous_bids,
                asks[0],
                self.previous_asks
            )[:self.mlofi_depth]
            if len(ml_ofi) < self.mlofi_depth:
                ml_ofi += [0.0] * (self.mlofi_depth - len(ml_ofi))
        ml_ofi = np.clip(np.array(ml_ofi) / self.parent_order_size, a_min=-1, a_max=1)

        # 11) Multi time OFI
        ofi_multi_time = []
        for t in range(1, min(len(bids), len(asks), self.ofi_lag + 1)):
            temp_ofi = markets_agent_utils.get_ml_ofi(
                bids[-t], bids[-(t + 1)],
                asks[-t], asks[-(t + 1)],
                depth=1
            )
            if len(temp_ofi) == 0:
                ofi_multi_time.append(0.0)
            else:
                ofi_multi_time.append(temp_ofi[0] / self.parent_order_size)

        # Pad OFI
        padded_ofi = np.zeros(self.ofi_lag, dtype=np.float32)
        last_ofi_len = len(ofi_multi_time)
        padded_ofi[-last_ofi_len:] = ofi_multi_time if last_ofi_len > 0 else padded_ofi
        padded_ofi = np.clip(padded_ofi, -1, 1)  # now bounded to [-1, 1] (but might clip a lot)

        # Set references for next step
        self.previous_bids = bids[0]
        self.previous_asks = asks[0]
        self.previous_depth = depth
        self.last_spread = spreads[-1]

        # 12) Build final state vector
        # Keep your enumerated structure, but use dimensionless/log scaled features
        computed_state = np.array([
                                      holdings_pct,
                                      scaled_mid_price,
                                      time_pct,
                                      diff_pct,
                                      imbalance_all,
                                      spread,
                                      short_term_vol,
                                      top_of_book_liquidity,
                                      depth
                                  ] + ml_ofi.tolist()
                                  + padded_ofi.tolist()
                                  + padded_returns.tolist(), dtype=np.float32)

        # Update step index
        self.step_index += 1

        return computed_state.reshape(self.num_state_features, 1)

    def compute_mid_price_ask_bid(self, raw_state: Dict[str, Any]):
        """
        Compute the mid_price, best_ask and best_bid
        Args:
            raw_state:

        Returns:    (mid_price, bid, ask)

        """
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        mid_prices = [
            markets_agent_utils.get_mid_price(bid, ask, lt)
            for (bid, ask, lt) in zip(bids, asks, last_transactions)
        ]
        mid_price = mid_prices[-1]  # current mid

        # 6) Best bids / asks
        best_bids = [
            b[0][0] if len(b) > 0 else mp
            for (b, mp) in zip(bids, mid_prices)
        ]
        best_asks = [
            a[0][0] if len(a) > 0 else mp
            for (a, mp) in zip(asks, mid_prices)
        ]

        best_bid = best_bids[-1]
        best_ask = best_asks[-1]
        mid_price = (best_bid + best_ask) / 2

        return mid_price, best_bid, best_ask

    #
    # ────────────────────────────────────────────────────────────────────────────────
    #  Reward helpers
    # ────────────────────────────────────────────────────────────────────────────────
    #

    # ────────────────────────────────────────────────────────────────────────────────
    #  Helper: Trading-PnL per step
    # ────────────────────────────────────────────────────────────────────────────────
    def compute_tpt(self, raw_state: Dict[str, Any], mid_price: float | None = None) -> float:
        """
        TP_t (Trading-PnL):
            Σ   qty   · (mid_price − fill_price)
            qty  > 0  for BUY fills,
            qty  < 0  for SELL fills.
        Returns the *raw* value (no weighting, no scaling).
        """
        if mid_price is None:
            mid_price = self.last_mid_price

        executed = raw_state["internal_data"]["inter_wakeup_executed_orders"]
        if not executed:
            return 0.0

        fills = executed[-1] if isinstance(executed[-1], list) else executed
        tp_raw = 0.0
        for fill in fills:
            qty_signed = fill.quantity if fill.side.value == "BID" else -fill.quantity
            tp_raw += qty_signed * (mid_price - fill.fill_price)
        return tp_raw

    def compute_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        Produce raw reward components; external ComponentNormalizer
        will scale & combine them.  This method now returns 0.
        """
        # --- market snapshot ------------------------------------------
        holdings = raw_state["internal_data"]["holdings"]
        cash = raw_state["internal_data"]["cash"]
        mid_px = self.last_mid_price
        value_t = cash + holdings * mid_px

        # ΔPnL (no window here – the wrapper can still damp / scale)
        dp_raw = value_t - getattr(self, "previous_marked_to_market", value_t)
        self.previous_marked_to_market = value_t

        # Trading PnL ---------------------------------------------------
        tp_raw = self.compute_tpt(raw_state, mid_px)

        # Inventory / quote-centre costs --------------------------------
        ip_raw = self.inventory_penalty * holdings ** 2
        res_px = 0.5 * (self.last_bid_action + self.last_ask_action)
        target_res = mid_px - self.kappa * holdings
        cp_raw = self.lambda_r * (res_px - target_res) ** 2

        # Fill-ratio bonus ---------------------------------------------
        fills = raw_state["internal_data"]["inter_wakeup_executed_orders"]
        qty = 0
        if fills:
            latest = fills[-1] if isinstance(fills[-1], list) else fills
            qty = sum(f.quantity for f in latest)
        fr_raw = self.fill_ratio_bonus * (qty / (2 * self.order_fixed_size) - 0.5)

        # Spread cliff penalties (kept raw) -----------------------------
        mkt_bid = mid_px - self.last_spread / 2
        mkt_ask = mid_px + self.last_spread / 2
        buf = 2

        bid_pen = (
            math.exp(max(0.0, (mkt_bid - buf) - self.last_bid_action) - 4.0)
            * self.out_of_spread_error
            if self.last_bid_action < mkt_bid - buf else 0.0
        )
        ask_pen = (
            math.exp(max(0.0, self.last_ask_action - (mkt_ask + buf)) - 4.0)
            * self.out_of_spread_error
            if self.last_ask_action > mkt_ask + buf else 0.0
        )

        # --- expose raw components to wrapper & callback --------------
        comps = {
            "DP": dp_raw,
            "TP": tp_raw,
            "FR": fr_raw,
            "IP": ip_raw,
            "CP": cp_raw,
            "CL": bid_pen + ask_pen  # treat cliff as one cost component
        }
        self.custom_metrics_tracker.dp_t = dp_raw
        self.custom_metrics_tracker.tp_t = tp_raw
        self.custom_metrics_tracker.fr_t = fr_raw
        self.custom_metrics_tracker.ip_t = ip_raw
        self.custom_metrics_tracker.cp_t = cp_raw
        self.custom_metrics_tracker.bid_penalty = bid_pen
        self.custom_metrics_tracker.ask_penalty = ask_pen
        self.custom_metrics_tracker.reward_components = comps

        # env returns 0; wrapper adds true reward
        return 0.0

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        return self.compute_reward(raw_state)

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        method that transforms a raw state into the final step reward update (if needed)

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: update reward computed at the end of the episode for the execution v0 environnement
        """
        # can update with additional reward at end of episode depending on scenario normalized by parent_order_size

        # 1) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 2) parent_order_size
        parent_order_size = self.parent_order_size

        update_reward = 0

        # # 3) Compute update_reward
        # if holdings >= 30 * parent_order_size:
        #     pass
        #     # update_reward = - 500_000  # executed buy too much
        # elif holdings <= - 30 * parent_order_size:
        #     pass
        #     # update_reward = - 500_000  # executed sell too much
        # else:
        #     update_reward = self.just_quantity_reward_update

        current_time = raw_state["internal_data"]["current_time"]
        mkt_open = raw_state["internal_data"]["mkt_open"]
        time_limit = mkt_open + self.first_interval + self.execution_window

        # Penalized if the agent does not finish the episode (not sure if this works though)
        # if current_time <= time_limit:
        #     update_reward -= 100_000 * (time_limit - current_time) / current_time

        update_reward -= self.terminal_inventory_penalty * (holdings ** 3) / self.parent_order_size

        # ------------ NEW: expose as component ---------------------------
        self.custom_metrics_tracker.reward_components["TI"] = update_reward

        # 4) Normalization
        update_reward = update_reward

        self.custom_metrics_tracker.late_penalty_reward = update_reward
        return update_reward

    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        """
        method that transforms a raw state into the flag if an episode is done

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - done: flag that describes if the episode is terminated or not  for the execution v0 environnement
        """
        # episode can stop because market closes or because some condition is met
        # here the condition is parent order fully executed

        # 1) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 2) parent_order_size
        parent_order_size = self.parent_order_size

        # 3) current time
        current_time = raw_state["internal_data"]["current_time"]

        # 4) time_limit
        # 4)a) mkt_open
        mkt_open = raw_state["internal_data"]["mkt_open"]
        # 4)b time_limit
        time_limit = mkt_open + self.first_interval + self.execution_window

        # 5) conditions
        if holdings >= parent_order_size:
            done = True  # Buy parent order executed
        elif holdings <= - parent_order_size:
            done = True  # Sell parent order executed
        elif current_time >= time_limit:
            done = True  # Mkt Close
        else:
            done = False

        self.custom_metrics_tracker.executed_quantity = (
            holdings if self.direction == "BUY" else -holdings
        )
        self.custom_metrics_tracker.remaining_quantity = (
                parent_order_size - self.custom_metrics_tracker.executed_quantity
        )

        return done

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        method that transforms a raw state into an info dictionnary

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: info dictionnary computed at each step for the execution v0 environnement
        """
        # Agent cannot use this info for taking decision
        # only for debugging

        # 1) Last Known Market Transaction Price
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]

        # 2) Last Known best bid
        bids = raw_state["parsed_mkt_data"]["bids"]
        best_bid = bids[0][0] if len(bids) > 0 else last_transaction

        # 3) Last Known best ask
        asks = raw_state["parsed_mkt_data"]["asks"]
        best_ask = asks[0][0] if len(asks) > 0 else last_transaction

        # 4) Current Time
        current_time = raw_state["internal_data"]["current_time"]

        # 5) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 6) Volume on both sides
        volume_bid = raw_state["parsed_volume_data"]["bid_volume"]
        volume_ask = raw_state["parsed_volume_data"]["ask_volume"]
        total_volume = volume_ask + volume_bid

        # 7) Cash
        cash = raw_state["internal_data"]["cash"]

        reward = self.custom_metrics_tracker.reward
        reserv = None
        if self.direct_action:
            bid_price, ask_price, _ = self.compute_bid_ask_direct(self.last_action_1, self.last_action_2)
        else:
            bid_price, ask_price, reserv = self.compute_bid_ask_reservation(self.last_action_1, self.last_action_2,
                                                                            extra_info=True)
        spread_width = ask_price - bid_price
        market_spread = self.last_spread

        if self.debug_mode:
            return {
                "last_transaction": last_transaction,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "cash_pct": cash / self.starting_cash,
                "volume_bid": volume_bid,
                "volume_ask": volume_ask,
                "total_volume": total_volume,
                "mid_price": (best_bid + best_ask) / 2,
                "current_time": current_time,
                "holdings": holdings,
                "parent_size": self.parent_order_size,
                "mtm_pnl": self.previous_marked_to_market - self.starting_cash,
                "bid_penalty": self.custom_metrics_tracker.bid_penalty,
                "ask_penalty": self.custom_metrics_tracker.ask_penalty,
                "tp_t": self.custom_metrics_tracker.tp_t,
                "dp_t": self.custom_metrics_tracker.dp_t,
                "ip_t": self.custom_metrics_tracker.ip_t,
                "fr_t": self.custom_metrics_tracker.fr_t,
                "cp_t": self.custom_metrics_tracker.cp_t,
                "reward": reward,
                "spread_ac": self.last_action_1,
                "reservation_ac": self.last_action_2,
                "reserv_price": abs(reserv) if reserv else self.last_mid_price,
                "spread_width": spread_width,
                "market_spread": market_spread,
                "reward_components": self.custom_metrics_tracker.reward_components
            }
        else:
            return asdict(self.custom_metrics_tracker)
