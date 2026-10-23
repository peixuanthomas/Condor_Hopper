import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from condor_hopper import cli


@pytest.fixture
def runtime(monkeypatch):
    connection, vehicle = Mock(), Mock()
    connect, fly = Mock(return_value=connection), Mock(return_value=0)
    monkeypatch.setitem(sys.modules, "krpc", SimpleNamespace(connect=connect))
    monkeypatch.setattr(cli, "KrpcVehicle", Mock(return_value=vehicle))
    monkeypatch.setattr(cli, "fly", fly)
    return SimpleNamespace(connection=connection, vehicle=vehicle, connect=connect, fly=fly)


@pytest.mark.parametrize("status", [0, 2])
def test_fly_runs_default_10000m_without_an_execute_flag(runtime, status):
    runtime.fly.return_value = status
    assert cli.main(["fly"]) == status
    runtime.fly.assert_called_once_with(runtime.vehicle, 10000.0)
    runtime.connect.assert_called_once_with(name="Condor Hopper", address="127.0.0.1",
        rpc_port=50000, stream_port=50001, use_pregenerated_stubs=False)
    runtime.connection.close.assert_called_once()


def test_fly_accepts_custom_altitude_and_connection(runtime):
    assert cli.main(["fly", "--target-altitude", "20000", "--address", "192.0.2.1",
                     "--rpc-port", "51000", "--stream-port", "51001"]) == 0
    runtime.fly.assert_called_once_with(runtime.vehicle, 20000.0)
    assert runtime.connect.call_args.kwargs["address"] == "192.0.2.1"
    assert runtime.connect.call_args.kwargs["rpc_port"] == 51000
    assert runtime.connect.call_args.kwargs["stream_port"] == 51001


def test_probe_prints_status_without_starting_flight(runtime, capsys):
    runtime.vehicle.snapshot.return_value = {"name": "Condor-Hopper", "mass_kg": 57494.0}
    assert cli.main(["probe"]) == 0
    assert json.loads(capsys.readouterr().out) == runtime.vehicle.snapshot.return_value
    runtime.fly.assert_not_called()
    runtime.connection.close.assert_called_once()


@pytest.mark.parametrize("arguments", [[], ["--help"], ["fly", "--help"], ["probe", "--help"]])
def test_help_works_without_connecting(runtime, capsys, arguments):
    if arguments:
        with pytest.raises(SystemExit) as result:
            cli.main(arguments)
        assert result.value.code == 0
    else:
        assert cli.main(arguments) == 0
    assert "condor-hopper" in capsys.readouterr().out
    runtime.connect.assert_not_called()


@pytest.mark.parametrize("altitude", ["nan", "inf", "0", "-1"])
def test_invalid_cutoff_rejected_before_connecting(runtime, capsys, altitude):
    with pytest.raises(SystemExit) as result:
        cli.main(["fly", "--target-altitude", altitude])
    assert result.value.code == 2
    assert "有限数值" in capsys.readouterr().err
    runtime.connect.assert_not_called()


def test_connection_closes_when_flight_fails(runtime):
    runtime.fly.side_effect = RuntimeError("flight error")
    with pytest.raises(RuntimeError, match="flight error"):
        cli.main(["fly"])
    runtime.connection.close.assert_called_once()
