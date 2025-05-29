import pandas as pd
from datetime import date

from .ExecutionAgent import ExecutionAgent
from ..utils import log_print


class VWAPExecutionAgent(ExecutionAgent):
    """
    VWAP (Volume Weighted Average Price) Execution Agent:
        breaks up the parent order and releases dynamically smaller chunks of the order to the market using
        stock specific historical volume profiles. Aims to execute the parent order as close as possible to the
        VWAP average price
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 direction, quantity, execution_time_horizon, freq, volume_profile_path,
                 trade=True, log_orders=False, random_state=None):
        super().__init__(id, name, type, symbol, starting_cash,
                         direction=direction, quantity=quantity, execution_time_horizon=execution_time_horizon,
                         trade=trade, log_orders=log_orders, random_state=random_state)
        if freq.endswith('m'):
            self.freq = freq.split('m')[0] + 'min'
        else:
            self.freq = freq
        self.volume_profile_path = volume_profile_path
        self.schedule = self.generate_schedule()

    def generate_schedule(self):

        if self.volume_profile_path is None:
            volume_profile = VWAPExecutionAgent.synthetic_volume_profile(self.freq, self.start_time)
        else:
            volume_profile = pd.read_pickle(self.volume_profile_path).to_dict()

        schedule = {}
        bins = pd.interval_range(start=pd.to_datetime(self.start_time), end=pd.to_datetime(self.end_time), freq=self.freq)
        for b in bins:
            schedule[b] = round(volume_profile[b.left] * self.quantity)
        log_print('[---- {} {} - Schedule ----]:'.format(self.name, self.current_time))
        log_print('[---- {} {} - Total Number of Orders ----]: {}'.format(self.name, self.current_time, len(schedule)))
        for t, q in schedule.items():
            log_print("From: {}, To: {}, Quantity: {}".format(t.left.time(), t.right.time(), q))
        return schedule

    @staticmethod
    def synthetic_volume_profile(freq, start_time):
        today = pd.to_datetime(start_time).normalize()
        mkt_open = today + pd.to_timedelta('09:30:00')
        mkt_close = today + pd.to_timedelta('16:00:00')
        day_range = pd.date_range(mkt_open, mkt_close, freq=freq)

        vol_profile = {}
        for t, x in zip(day_range, range(int(-len(day_range) / 2), int(len(day_range) / 2), 1)):
            vol_profile[t] = x ** 2 + 2 * x + 2

        factor = 1.0 / sum(vol_profile.values())
        vol_profile = {k: v * factor for k, v in vol_profile.items()}

        return vol_profile