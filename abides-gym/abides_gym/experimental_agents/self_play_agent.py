from collections import deque
from copy import deepcopy

import numpy as np
from typing import Optional, List, Tuple, Dict, Any

import stable_baselines3.common.base_class
import torch.nn
from abides_core import NanosecondTime
from abides_core.generators import InterArrivalTimeGenerator, ConstantTimeGenerator
from abides_core.utils import str_to_ns
from abides_markets.orders import Order

from .self_core_background_agent import SelfCoreBackgroundAgent
import abides_markets.agents.utils as markets_agent_utils


class SelfPlayAgent(SelfCoreBackgroundAgent):
    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )

    def __init__(
            self,
            id: int,
            symbol: str,
            starting_cash: int,
            nn_model: stable_baselines3.common.base_class.BaseAlgorithm,
            environment_configuration: Dict[str, Any],
            subscribe_freq: int = int(1e8),
            subscribe: float = True,
            subscribe_num_levels: int = 10,
            wakeup_interval_generator: InterArrivalTimeGenerator = ConstantTimeGenerator(
                step_duration=str_to_ns("1min")
            ),
            state_buffer_length: int = 2,
            market_data_buffer_length: int = 5,
            first_interval: Optional[NanosecondTime] = None,
            log_orders: bool = False,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(
            id,
            symbol=symbol,
            starting_cash=starting_cash,
            log_orders=log_orders,
            name=name,
            type=type,
            random_state=random_state,
            wakeup_interval_generator=wakeup_interval_generator,
            state_buffer_length=state_buffer_length,
            market_data_buffer_length=market_data_buffer_length,
            first_interval=first_interval,
            subscribe=subscribe,
            subscribe_num_levels=subscribe_num_levels,
            subscribe_freq=subscribe_freq,
        )
        self.last_mid_price = None
        self.symbol: str = symbol
        # Frequency of agent data subscription up in ns-1
        self.subscribe_freq: int = subscribe_freq
        self.subscribe: bool = subscribe
        self.subscribe_num_levels: int = subscribe_num_levels

        self.wakeup_interval_generator: InterArrivalTimeGenerator = (
            wakeup_interval_generator
        )
        self.lookback_period: NanosecondTime = self.wakeup_interval_generator.mean()

        if hasattr(self.wakeup_interval_generator, "random_generator"):
            self.wakeup_interval_generator.random_generator = self.random_state

        self.state_buffer_length: int = state_buffer_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: Optional[NanosecondTime] = first_interval
        # internal variables
        self.has_subscribed: bool = False
        self.episode_executed_orders: List[
            Order
        ] = []  # list of executed orders during full episode

        # list of executed orders between steps - is reset at every step
        self.inter_wakeup_executed_orders: List[Order] = []
        self.parsed_episode_executed_orders: List[Tuple[int, int]] = []  # (price, qty)
        self.parsed_inter_wakeup_executed_orders: List[
            Tuple[int, int]
        ] = []  # (price, qty)
        self.parsed_mkt_data: Dict[str, Any] = {}
        self.parsed_mkt_data_buffer = deque(maxlen=self.market_data_buffer_length)
        self.parsed_volume_data = {}
        self.parsed_volume_data_buffer = deque(maxlen=self.market_data_buffer_length)
        self.raw_state = deque(maxlen=self.state_buffer_length)
        # dictionary to track order status:
        # - keys = order_id
        # - value = dictionary {'active'|'cancelled'|'executed', Order, 'active_qty','executed_qty', 'cancelled_qty }
        self.order_status: Dict[int, Dict[str, Any]] = {}

        # Very important, this contains all the previously trained on metrics for the agent on which he learned
        self.environment_configuration = environment_configuration

        # Extract the relevant metrics out for the agents
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

        self.previous_asks: list | None = None
        self.previous_bids: list | None = None
        self.previous_depth: int = 0

        # Load the model here
        self.model: stable_baselines3.common.base_class.BaseAlgorithm = nn_model

    def act_on_wakeup(self) -> Dict:
        """
        Computes next wakeup time, computes the new raw_state and clears the internal step buffers.
        Returns the raw_state to the abides gym environment (outside of the abides simulation) where the next action will be selected.

        Returns:
            - the raw_state dictionary that will be processed in the abides gym subenvironment
        """
        # compute the state (returned to the Gym Env)
        # wakeup logic
        wake_time = (
                self.current_time + self.wakeup_interval_generator.next()
        )  # generates next wakeup time
        self.set_wakeup(wake_time)
        self.update_raw_state()
        raw_state = deepcopy(self.get_raw_state())
        self.new_step_reset()
        # return non None value so the kernel catches it and stops
        return raw_state

    def submit_actions(self) -> None:
        # 1. Get the raw_state (make sure its updated beforehand)
        raw_state = self.get_raw_state()

        if len(raw_state) > 0:

            # 2. Transform the raw_state to something meaningful
            state = self.raw_state_to_state(raw_state)

            # 3. Forward the processed the raw_state through the Neural Network
            action_list, _ = self.model.predict(state)

            # 4. Get the actions out of the neural network
            actions = self.convert_actions_to_abides(action_list.tolist())

            # 5. Forward the actions through the action mapper
            self.apply_actions(actions)

    def compute_bid_ask_reservation(self, spread_val, res_val, extra_info=False) -> tuple[int, int, float | None]:
        """
        Function for calculating the bid-ask spread based on the input
        Args:
            spread_val: x in (0, 1) to indicate how wide the spread will be
            res_val: y in (0, 1) to indicate how far off the mid-price we will quote our mid-price

        Returns: The bid-price and ask-price that will be quoted

        """
        mid_price = self.last_mid_price  # ~100,000

        # Reservation Price
        reservation_price = mid_price - self.reservation_quote * mid_price * (2 * res_val - 1)

        # Spread (split in half around the reservation_price)
        spread_val = min(max((spread_val + 1) / 2, 0), 1)
        half_spread = (spread_val * self.max_spread * mid_price) / 2.0

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
        mid_price = self.last_mid_price  # ~100,000

        bid_price = round(mid_price * (1 - bid_val))
        ask_price = round(mid_price * (1 + ask_val))

        return bid_price, ask_price, None

    def convert_actions_to_abides(self, action: list):
        """
        action is a 4D float array:
          [bid_offset, ask_offset, bid_size_float, ask_size_float]

        We'll round the size floats to integers for the order instructions.
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
        holdings_pct = holdings[-1] / self.parent_order_size  # dimensionless in [-1,1]

        # 2) Timing
        mkt_open = raw_state["internal_data"]["mkt_open"][-1]
        current_time = raw_state["internal_data"]["current_time"][-1]
        time_from_parent_arrival = current_time - mkt_open - self.first_interval

        # assert (current_time >= mkt_open + self.first_interval), (
        #     "Agent has woken up earlier than its first interval"
        # )

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
        scaled_mid_price = self.last_mid_price / self.scale_price  # dimensionless around -1

        # Spread as fraction of mid
        spreads = np.array(best_asks) - np.array(best_bids)
        spread = spreads[-1] / self.last_mid_price  # dimensionless relative to mid_price < 1/2

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

        return computed_state.reshape(self.num_state_features, 1)
