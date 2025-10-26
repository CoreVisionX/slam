"""
Download the tartanair datasets used for testing
"""

import tartanair as ta

# Initialize TartanAir.
tartanair_data_root = 'tests/data'
ta.init(tartanair_data_root)

# Download the datasets.
ta.download_multi_thread(config='tests/config/tartanair.yaml', num_workers=16)