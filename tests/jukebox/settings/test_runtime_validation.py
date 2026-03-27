from types import SimpleNamespace

import pytest

from jukebox.settings.runtime_validation import validate_resolved_jukebox_runtime_rules
from jukebox.settings.value_providers import ObjectLeafValueProvider
from tests.jukebox.settings._helpers import build_resolved_sonos_group_runtime


def test_validate_resolved_jukebox_runtime_rules_allows_dryrun_without_sonos_target():
    provider = ObjectLeafValueProvider(
        SimpleNamespace(
            player_type="dryrun",
            sonos_host=None,
            sonos_name=None,
            sonos_group=None,
            loop_interval_seconds=0.1,
            pause_delay_seconds=0.25,
        )
    )

    validate_resolved_jukebox_runtime_rules(provider)


def test_validate_resolved_jukebox_runtime_rules_allows_missing_sonos_target_for_autodiscovery():
    provider = ObjectLeafValueProvider(
        SimpleNamespace(
            player_type="sonos",
            sonos_host=None,
            sonos_name=None,
            sonos_group=None,
            loop_interval_seconds=0.1,
            pause_delay_seconds=0.25,
        )
    )

    validate_resolved_jukebox_runtime_rules(provider)


def test_validate_resolved_jukebox_runtime_rules_rejects_group_host_mismatch():
    provider = ObjectLeafValueProvider(
        SimpleNamespace(
            player_type="sonos",
            sonos_host="192.168.1.99",
            sonos_name=None,
            sonos_group=build_resolved_sonos_group_runtime(),
            loop_interval_seconds=0.1,
            pause_delay_seconds=0.25,
        )
    )

    with pytest.raises(ValueError, match="coordinator host"):
        validate_resolved_jukebox_runtime_rules(provider)
