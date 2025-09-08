import os
import tempfile
import pytest

from scripts.state_utils import StopToken, StopRequested, STOP_FILENAME


def test_stop_token_manual_set_and_file_triggers():
    with tempfile.TemporaryDirectory() as d:
        st = StopToken(d)
        # Initially not set
        assert not st.is_set()

        # Manual set
        st.set()
        assert st.is_set()
        with pytest.raises(StopRequested):
            st.check()

        # Reset by recreating token and using STOP file
        st2 = StopToken(d)
        stop_path = os.path.join(d, STOP_FILENAME)
        with open(stop_path, "w", encoding="utf-8") as f:
            f.write("test")
        assert st2.is_set()  # file presence should flip the flag
        with pytest.raises(StopRequested):
            st2.check()

