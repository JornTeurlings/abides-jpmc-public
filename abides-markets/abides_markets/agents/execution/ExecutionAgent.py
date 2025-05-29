import pandas as pd
import datetime

from abides_markets.orders import Side
from abides_markets.messages.query import QuerySpreadMsg, QuerySpreadResponseMsg

from ..trading_agent import TradingAgent
from ..utils import log_print


class ExecutionAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, starting_cash,
                 direction, quantity, execution_time_horizon,
                 trade=True, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.execution_time_horizon = execution_time_horizon

        self.start_time = self.execution_time_horizon[0]
        self.end_time = self.execution_time_horizon[-1]
        self.schedule = None

        self.rem_quantity = quantity
        self.arrival_price = None

        self.accepted_orders = []
        self.trade = trade
        self.log_orders = log_orders

        self.state = 'AWAITING_WAKEUP'

    def kernel_stopping(self):
        super().kernel_stopping()
        if self.trade:
            slippage = self.get_average_transaction_price() - self.arrival_price if self.direction == 'BUY' else \
                self.arrival_price - self.get_average_transaction_price()
            self.logEvent('DIRECTION', self.direction, True)
            self.logEvent('TOTAL_QTY', self.quantity, True)
            self.logEvent('REM_QTY', self.rem_quantity, True)
            self.logEvent('ARRIVAL_MID', self.arrival_price, True)
            self.logEvent('AVG_TXN_PRICE', self.get_average_transaction_price(), True)
            self.logEvent('SLIPPAGE', slippage, True)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if not can_trade: return
        if self.trade:
            try:
                self.set_wakeup([int(time.to_datetime64()) for time in self.execution_time_horizon if int(time.to_datetime64()) > currentTime][0])
            except IndexError:
                pass

            self.get_current_spread(self.symbol, depth=1000)
            self.state = 'AWAITING_SPREAD'

    def receive_message(self, current_time, sender_id, msg):
        super().receive_message(current_time, sender_id, msg)

        if (self.rem_quantity > 0 and
                self.state == 'AWAITING_SPREAD' and
                isinstance(msg, QuerySpreadResponseMsg)):
            self.cancel_all_orders()
            self.placeOrders(current_time)
            self.state = "AWAITING_WAKEUP"

    def order_executed(self, order):
        executed_order = order
        self.executed_orders.append(executed_order)
        executed_qty = sum(executed_order.quantity for executed_order in self.executed_orders)
        self.rem_quantity = self.quantity - executed_qty
        log_print('[---- {} ----]: LIMIT ORDER EXECUTED - {} @ {}'.format(self.name, executed_order.quantity,
                                                                          executed_order.fill_price))
        log_print('[---- {} ----]: EXECUTED QUANTITY: {}'.format(self.name, executed_qty))
        log_print('[---- {} ----]: REMAINING QUANTITY: {}'.format(self.name, self.rem_quantity))
        log_print('[---- {} ----]: % EXECUTED: {} \n'.format(self.name,
                                                             round((1 - self.rem_quantity / self.quantity) * 100, 2)))

    def order_accepted(self, order):
        accepted_order = order
        self.accepted_orders.append(accepted_order)
        accepted_qty = sum(accepted_order.quantity for accepted_order in self.accepted_orders)
        log_print('[---- {} ----]: ACCEPTED QUANTITY : {}'.format(self.name, accepted_qty))

    def placeOrders(self, currentTime):
        currentTime = pd.Timestamp(currentTime)
        if currentTime.floor('1s') == self.execution_time_horizon[-2]:
            self.place_market_order(symbol=self.symbol, quantity=self.rem_quantity,
                                    side=Side.ASK if self.direction == 'ASK' else Side.BID)
        elif currentTime.floor('1s') in self.execution_time_horizon[:-2]:
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)

            if currentTime.floor('1s') == self.start_time:
                self.arrival_price = (bid + ask) / 2
                log_print("[---- {}  - {} ----]: Arrival Mid Price {}".format(self.name, currentTime,
                                                                              self.arrival_price))

            qty = self.schedule[pd.Interval(currentTime.floor('1s'),
                                            currentTime.floor('1s') + datetime.timedelta(minutes=30))]
            price = ask if self.direction == 'BUY' else bid
            self.place_limit_order(symbol=self.symbol, quantity=qty,
                                   side=Side.ASK if self.direction == 'ASK' else Side.BID, limit_price=price)
            log_print('[---- {} - {} ----]: LIMIT ORDER PLACED - {} @ {}'.format(self.name, currentTime, qty, price))

    def cancel_all_orders(self):
        for _, order in self.orders.items():
            self.cancel_order(order)

    def get_wake_frequency(self):
        return pd.Timedelta(self.execution_time_horizon[1] - self.execution_time_horizon[0]).seconds * 1e9
