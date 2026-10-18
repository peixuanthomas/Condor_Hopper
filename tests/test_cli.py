import pytest

import main


def test_flight_requires_execute_before_connecting():
    with pytest.raises(SystemExit, match="--execute"):
        main.main(["fly"])


def test_default_cutoff_is_10000m():
    assert main.build_parser().parse_args(["fly"]).target_altitude == 10000.0


@pytest.mark.parametrize("altitude", ["nan", "inf", "0", "-1"])
def test_invalid_cutoff_rejected_before_connecting(altitude):
    with pytest.raises(SystemExit, match="有限数值"):
        main.main(["fly", "--execute", "--target-altitude", altitude])
