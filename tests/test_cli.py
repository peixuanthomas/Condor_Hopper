import pytest

import main


def test_live_flight_requires_explicit_execute_flag():
    args = main.build_parser().parse_args(["fly"])

    with pytest.raises(SystemExit, match="--execute"):
        main.run_flight(args)


def test_hover_option_was_removed_from_mission_cli():
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["simulate", "--hover-time", "10"])
