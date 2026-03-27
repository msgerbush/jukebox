from .validation_rules import SettingsValidationRule, validate_settings_rules
from .value_providers import SettingsValueProvider


def validate_resolved_jukebox_runtime_rules(provider: SettingsValueProvider) -> None:
    validate_settings_rules(provider)

    for rule in RUNTIME_VALIDATION_RULES:
        if all(provider.has_value(dependency_path) for dependency_path in rule.depends_on_paths):
            rule.validate(provider)


def validate_sonos_target_presence(
    player_type: str,
    sonos_host,
    sonos_name,
    sonos_group,
) -> None:
    if player_type != "sonos":
        return

    if sonos_host is None and sonos_name is None and sonos_group is None:
        raise ValueError("sonos player requires a resolved host, name, or group target")


def validate_sonos_group_runtime_consistency(
    player_type: str,
    sonos_host,
    sonos_name,
    sonos_group,
) -> None:
    if player_type != "sonos" or sonos_group is None:
        return

    if sonos_name is not None:
        raise ValueError("sonos_name cannot be set when sonos_group is present")

    if sonos_host != sonos_group.coordinator.host:
        raise ValueError("sonos_host must match the resolved Sonos group coordinator host")


RUNTIME_VALIDATION_RULES = (
    SettingsValidationRule(
        name="sonos_target_presence",
        depends_on_paths=(
            "runtime.player_type",
            "runtime.sonos_host",
            "runtime.sonos_name",
            "runtime.sonos_group",
        ),
        validator=validate_sonos_target_presence,
    ),
    SettingsValidationRule(
        name="sonos_group_runtime_consistency",
        depends_on_paths=(
            "runtime.player_type",
            "runtime.sonos_host",
            "runtime.sonos_name",
            "runtime.sonos_group",
        ),
        validator=validate_sonos_group_runtime_consistency,
    ),
)
