from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from ..price_level import PriceLevel


################## STATE MANIPULATION ###############################
def list_dict_flip(ld: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Utility that returns a dictionnary of list of dictionnary into a dictionary of list

    Arguments:
        - ld: list of dictionaary
    Returns:
        - flipped: dictionnary of lists
    Example:
        - ld = [{"a":1, "b":2}, {"a":3, "b":4}]
        - flipped = {'a': [1, 3], 'b': [2, 4]}
    """
    flipped = dict((k, []) for (k, v) in ld[0].items())
    for rs in ld:
        for k in flipped.keys():
            flipped[k].append(rs[k])
    return flipped


def identity_decorator(func):
    """
    identy for decorators: take a function and return that same function

    Arguments:
        - func: function
    Returns:
        - wrapper_identity_decorator: function
    """

    def wrapper_identity_decorator(*args, **kvargs):
        return func(*args, **kvargs)

    return wrapper_identity_decorator


def ignore_mkt_data_buffer_decorator(func):
    """
    Decorator for function that takes as input self and raw_state.
    Applies the given function while ignoring the buffering in the market data.
    Only last element of the market data buffer is kept
    Arguments:
        - func: function
    Returns:
        - wrapper_mkt_data_buffer_decorator: function
    """

    def wrapper_mkt_data_buffer_decorator(self, raw_state):
        raw_state_copy = deepcopy(raw_state)
        for i in range(len(raw_state)):
            raw_state[i]["parsed_mkt_data"] = raw_state_copy[i]["parsed_mkt_data"][-1]
            raw_state[i]["parsed_volume_data"] = raw_state_copy[i][
                "parsed_volume_data"
            ][-1]
        raw_state2 = list_dict_flip(raw_state)
        flipped = dict((k, list_dict_flip(v)) for (k, v) in raw_state2.items())
        return func(self, flipped)

    return wrapper_mkt_data_buffer_decorator


def ignore_buffers_decorator(func):
    """
    Decorator for function that takes as input self and raw_state.
    Applies the given function while ignoring the buffering in both the market data and the general raw state.
    Only last elements are kept.
    Arguments:
        - func: function
    Returns:
        - wrapper_mkt_data_buffer_decorator: function
    """

    def wrapper_ignore_buffers_decorator(self, raw_state):
        raw_state = raw_state[-1]
        if len(raw_state["parsed_mkt_data"]) == 0:
            pass
        else:
            raw_state["parsed_mkt_data"] = raw_state["parsed_mkt_data"][-1]
            if raw_state["parsed_volume_data"]:
                raw_state["parsed_volume_data"] = raw_state["parsed_volume_data"][-1]
        return func(self, raw_state)

    return wrapper_ignore_buffers_decorator


################# ORDERBOOK PRIMITIVES ######################
def get_mid_price(
    bids: List[PriceLevel], asks: List[PriceLevel], last_transaction: int
) -> int:

    """
    Utility that computes the mid price from the snapshot of bid and ask side

    Arguments:
        - bids: list of list snapshot of bid side
        - asks: list of list snapshot of ask side
        - last_trasaction: last transaction in the market, used for corner cases when one side of the OB is empty
    Returns:
        - mid_price value
    """
    if len(bids) == 0 and len(asks) == 0:
        return last_transaction
    elif len(bids) == 0:
        return asks[0][0]
    elif len(asks) == 0:
        return bids[0][0]
    else:
        return (bids[0][0] + asks[0][0]) / 2


def get_val(book: List[PriceLevel], level: int) -> Tuple[int, int]:
    """
    utility to compute the price and level at the level-th level of the order book

    Arguments:
        - book: side of the order book (bid or ask)
        - level: level of interest in the OB side (index starts at 0 for best bid/ask)

    Returns:
        - tuple price, volume for the i-th value
    """
    if book == []:
        return 0, 0
    else:
        try:
            price = book[level][0]
            volume = book[level][1]
            return price, volume
        except:
            return 0, 0


def get_last_val(book: List[PriceLevel], mid_price: int) -> int:
    """
    utility to compute the price of the deepest placed order in the side of the order book

    Arguments:
        - book: side of the order book (bid or ask)
        - mid_price: current mid price used for corner cases

    Returns:
        - mid price value
    """
    if book == []:
        return mid_price
    else:
        return book[-1][0]


def get_volume(book: List[PriceLevel], depth: Optional[int] = None) -> int:
    """
    utility to compute the volume placed between the top of the book (depth 0) and the depth

    Arguments:
        - book: side of the order book (bid or ask)
        - depth: depth used to compute sum of the volume

    Returns:
        - volume placed
    """
    if depth is None:
        return sum([v[1] for v in book])
    else:
        return sum([v[1] for v in book[:depth]])


def get_imbalance(
    bids: List[PriceLevel],
    asks: List[PriceLevel],
    depth: Optional[int] = None,
) -> float:
    """
    Computes the Queue Imbalance (QI), which measures how liquidity is distributed
    between the best bid and ask up to a given depth.

    Arguments:
        - bids: List of PriceLevel tuples [(price, size), ...] representing the bid side.
        - asks: List of PriceLevel tuples [(price, size), ...] representing the ask side.
        - depth: Depth up to which to sum the volume (default: top-of-book only).

    Returns:
        - QI: A float in the range [-1, 1], where:
            - Positive values (closer to +1) indicate higher liquidity on the bid side.
            - Negative values (closer to -1) indicate higher liquidity on the ask side.
            - 0 means perfectly balanced book at that depth.
    """
    # Handle edge cases where order book is empty
    if not bids and not asks:
        return 0.0
    elif not bids:
        return -1.0  # No bids, fully ask-dominated
    elif not asks:
        return 1.0   # No asks, fully bid-dominated

    # Sum volumes up to the specified depth (or all available levels if depth is None)
    if depth is None:
        bid_vol = sum(v[1] for v in bids)  # Summing all bid sizes
        ask_vol = sum(v[1] for v in asks)  # Summing all ask sizes
    else:
        bid_vol = sum(v[1] for v in bids[:depth])  # Summing bid sizes up to depth
        ask_vol = sum(v[1] for v in asks[:depth])  # Summing ask sizes up to depth

    # Avoid division by zero
    total_volume = bid_vol + ask_vol
    if total_volume == 0:
        return 0.0

    # Compute Queue Imbalance
    return (bid_vol - ask_vol) / total_volume


def get_ml_ofi(
        bids: List[PriceLevel],
        prev_bids: List[PriceLevel],
        asks: List[PriceLevel],
        prev_asks: List[PriceLevel],
        depth: Optional[int] = None
):
    """
    Computes Multi-Level Order Flow Imbalance (MLOFI) for multiple levels of the order book.

    Args:
        - bid_prices:
        - bid_sizes:
        - ask_prices:
        - ask_sizes:

    Returns:
        - List of lists containing MLOFI values for each level over time.
    """

    # We can only compute it upto a certain depth for which it is available
    if depth is None:
        max_depth = min(len(bids), len(prev_bids), len(asks), len(prev_asks))
    else:
        max_depth = min(len(bids), len(prev_bids), len(asks), len(prev_asks), depth)
    mlofi_values = []

    for m in range(max_depth):
        # Extract price and size for bid and ask at level m
        b_t, q_b_t = bids[m]  # Current best bid price and size
        b_t_1, q_b_t_1 = prev_bids[m]  # Previous best bid price and size

        a_t, q_a_t = asks[m]  # Current best ask price and size
        a_t_1, q_a_t_1 = prev_asks[m]  # Previous best ask price and size

        # Compute ΔW^m (Bid-side contribution)
        if b_t > b_t_1:
            delta_W = q_b_t
        elif b_t == b_t_1:
            delta_W = q_b_t - q_b_t_1
        else:  # b_t < b_t_1
            delta_W = -q_b_t_1

        # Compute ΔV^m (Ask-side contribution)
        if a_t > a_t_1:
            delta_V = -q_a_t_1
        elif a_t == a_t_1:
            delta_V = q_a_t - q_a_t_1
        else:  # a_t < a_t_1
            delta_V = q_a_t

        # Compute MLOFI for this level: e^m(τ_n) = ΔW^m(τ_n) - ΔV^m(τ_n)
        e_m = delta_W - delta_V
        mlofi_values.append(e_m)

    return mlofi_values


