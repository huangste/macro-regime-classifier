"""regimelab: market-regime classification and forecasting research library.

The package separates three concerns that the original notebook conflated:

* ``labeling``   -- descriptive, in-sample regime classification (legitimately
                    full-sample; it is a smoothing/description exercise).
* ``forecasting`` -- strictly causal prediction of a future regime state.
* ``evaluation``  -- purged/embargoed walk-forward validation against
                    baselines that already exploit regime persistence.
"""

__version__ = "0.1.0"
