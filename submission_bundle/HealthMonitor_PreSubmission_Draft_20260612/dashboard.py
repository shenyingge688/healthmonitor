"""Root entrypoint for the HealthMonitor Streamlit dashboard.

Streamlit reruns this file after every widget interaction. The actual dashboard
module renders at import time, so the module must be reloaded on rerun instead
of being pulled from Python's import cache.
"""

from importlib import import_module, reload
import sys


DASHBOARD_MODULE = "healthmonitor.dashboard"

if DASHBOARD_MODULE in sys.modules:
    reload(sys.modules[DASHBOARD_MODULE])
else:
    import_module(DASHBOARD_MODULE)
