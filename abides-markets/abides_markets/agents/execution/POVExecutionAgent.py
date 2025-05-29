import sys
import warnings
import pandas as pd
from abides_markets.messages.orderbook import OrderExecutedMsg, OrderAcceptedMsg
from abides_markets.messages.query import QueryTransactedVolMsg, QueryTransactedVolResponseMsg
from abides_markets.orders import Side

from ..trading_agent import TradingAgent
from ..utils import log_print

POVExecutionWarning_msg = "Running a configuration using POVExecutionAgent requires an ExchangeAgent with " \
                          "attribute `stream_history` set to a large value, recommended at sys.maxsize."


class POVExecutionAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, starting_cash,
                 direction, quantity, pov, start_time, freq, lookback_period, end_time=None,
                 trade=True, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.log_events = True  # save events for plotting
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.rem_quantity = quantity
        self.pov = pov
        self.start_time = start_time
        self.end_time = end_time
        self.freq = freq
        self.look_back_period = lookback_period
        self.trade = trade
        self.accepted_orders = []
        self.state = 'AWAITING_WAKEUP'

        warnings.warn(POVExecutionWarning_msg, UserWarning, stacklevel=1)
        self.processEndTime()

    def processEndTime(self):
        """ Make end time of POV order sensible, i.e. if a time is given leave it alone; else, add 24 hours to start."""
        if self.end_time is None:
            self.end_time = self.start_time + pd.to_timedelta('24 hours')

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if not can_trade:
            return
        if self.trade and self.rem_quantity > 0 and self.start_time <= currentTime < self.end_time:
            self.cancel_all_orders()
            self.get_current_spread(self.symbol, depth=sys.maxsize)
            self.get_transacted_volume(self.symbol, lookback_period=self.look_back_period)
            self.state = 'AWAITING_TRANSACTED_VOLUME'
            delta_time = self.get_wake_frequency()
            self.set_wakeup(currentTime + int(round(delta_time)))

    def get_wake_frequency(self):
        return pd.Timedelta(self.freq).seconds * 1e9

    def receive_message(self, currentTime, sender_id, msg):
        super().receive_message(currentTime, sender_id, msg)

        if currentTime > self.end_time:
            log_print(
                f'[---- {self.name} - {currentTime} ----]: current time {currentTime} is after specified end time of POV order '
                f'{self.end_time}. TRADING CONCLUDED. ')
            return

        if self.rem_quantity > 0 and \
                self.state == 'AWAITING_TRANSACTED_VOLUME' \
                and isinstance(msg, QueryTransactedVolResponseMsg) \
                and self.transacted_volume[self.symbol] is not None \
                and currentTime > self.start_time:
            qty = round(self.pov * self.transacted_volume[self.symbol][0 if self.direction == 'BUY' else 1])
            self.cancel_all_orders()
            self.place_market_order(self.symbol, qty, side=Side.BID if self.direction == 'BUY' else Side.ASK)
            log_print(
                f'[---- {self.name} - {currentTime} ----]: TOTAL TRANSACTED VOLUME IN THE LAST {self.look_back_period} = {self.transacted_volume[self.symbol]}')
            log_print(f'[---- {self.name} - {currentTime} ----]: MARKET ORDER PLACED - {qty}')
            self.state = "AWAITING_WAKEUP"

    def order_accepted(self, order):
        accepted_order = order
        self.accepted_orders.append(accepted_order)
        accepted_qty = sum(accepted_order.quantity for accepted_order in self.accepted_orders)
        log_print(f'[---- {self.name} ----]: ACCEPTED QUANTITY : {accepted_qty}')

    def order_executed(self, order):
        executed_order = order
        self.executed_orders.append(executed_order)
        executed_qty = sum(executed_order.quantity for executed_order in self.executed_orders)
        self.rem_quantity = self.quantity - executed_qty
        log_print(
            f'[---- {self.name} ----]: LIMIT ORDER EXECUTED - {executed_order.quantity} @ {executed_order.fill_price}')
        log_print(f'[---- {self.name} ----]: EXECUTED QUANTITY: {executed_qty}')
        log_print(f'[---- {self.name} ----]: REMAINING QUANTITY (NOT EXECUTED): {self.rem_quantity}')
        log_print(
            f'[---- {self.name} ----]: % EXECUTED: {round((1 - self.rem_quantity / self.quantity) * 100, 2)} \n')

    def cancel_all_orders(self):
        for _, order in self.orders.items():
            self.cancel_order(order)
