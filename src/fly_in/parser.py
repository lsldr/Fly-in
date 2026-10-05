import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from fly_in.models import (
    EdgeZoneSpec,
    MapConfig,
    RegZoneSpec,
    RouteSpec,
)
from fly_in.errors import ConfigParseError

ModelT = TypeVar("ModelT", bound=BaseModel)


class Parser:
    """Parse map files into validated, simulation-ready configuration."""

    def parse(self, filename: str | Path) -> MapConfig:
        """Read and validate a map configuration file."""
        try:
            text = Path(filename).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ConfigParseError(
                f"cannot read {filename}: {error}"
            ) from error
        return self.parse_text(text)

    def parse_text(self, text: str) -> MapConfig:
        """Parse configuration text, reporting errors against source lines."""
        records = [
            (line_number, content.split("#", 1)[0].strip())
            for line_number, content in enumerate(text.splitlines(), start=1)
        ]
        records = [(number, line) for number, line in records if line]
        if not records:
            raise ConfigParseError("line 1: configuration is empty")

        first_line_number, first_line = records[0]
        drone_match = re.fullmatch(r"nb_drones:\s*([0-9]+)\s*", first_line)
        if not drone_match:
            raise ConfigParseError(
                f"line {first_line_number}: expected "
                "'nb_drones: <positive_integer>'"
            )
        try:
            nb_drones = int(drone_match.group(1))
            if nb_drones < 1:
                raise ValueError
        except ValueError as error:
            raise ConfigParseError(
                f"line {first_line_number}: nb_drones must be a "
                "positive integer"
            ) from error

        start: EdgeZoneSpec | None = None
        end: EdgeZoneSpec | None = None
        zones: list[RegZoneSpec] = []
        connections: list[RouteSpec] = []
        defined_names: set[str] = set()
        connection_pairs: set[frozenset[str]] = set()

        for line_number, line in records[1:]:
            try:
                prefix, separator, body = line.partition(":")
                if not separator or prefix not in {
                    "start_hub",
                    "end_hub",
                    "hub",
                    "connection",
                }:
                    raise ConfigParseError(
                        f"line {line_number}: expected start_hub:, end_hub:, "
                        "hub:, or connection:"
                    )
                body = body.strip()
                primary, raw_metadata = self._split_metadata(
                    body, line_number, 1 if prefix == "connection" else 3
                )
                values = self._parse_metadata(raw_metadata, line_number)

                if prefix == "connection":
                    route = self._parse_connection(
                        primary, values, line_number, defined_names
                    )
                    pair = frozenset((route.from_hub, route.to_hub))
                    if pair in connection_pairs:
                        raise ConfigParseError(
                            f"line {line_number}: duplicate connection "
                            f"'{route.from_hub}-{route.to_hub}'"
                        )
                    connection_pairs.add(pair)
                    connections.append(route)
                    continue

                zone_name, x, y = self._parse_zone_coordinates(
                    primary, line_number
                )
                if zone_name in defined_names:
                    raise ConfigParseError(
                        f"line {line_number}: zone name '{zone_name}' is "
                        "already defined"
                    )

                zone_values: dict[str, Any] = {
                    "name": zone_name,
                    "x": x,
                    "y": y,
                }
                if "zone" in values:
                    zone_values["zone_type"] = values["zone"]
                if "color" in values:
                    zone_values["color"] = values["color"]

                if prefix in {"start_hub", "end_hub"}:
                    # Endpoint capacity is ignored.
                    invalid_fields = set(values) - {
                        "zone",
                        "color",
                        "max_drones",
                    }
                    if invalid_fields:
                        field = sorted(invalid_fields)[0]
                        raise ConfigParseError(
                            f"line {line_number}: metadata field '{field}' "
                            "is not valid for a zone"
                        )
                    endpoint_zone = self._validate_model(
                        EdgeZoneSpec, zone_values, line_number, "zone"
                    )
                    if prefix == "start_hub":
                        if start is not None:
                            raise ConfigParseError(
                                f"line {line_number}: start_hub is defined "
                                "more than once"
                            )
                        start = endpoint_zone
                    else:
                        if end is not None:
                            raise ConfigParseError(
                                f"line {line_number}: end_hub is defined "
                                "more than once"
                            )
                        end = endpoint_zone
                else:
                    invalid_fields = set(values) - {
                        "zone",
                        "color",
                        "max_drones",
                    }
                    if invalid_fields:
                        field = sorted(invalid_fields)[0]
                        raise ConfigParseError(
                            f"line {line_number}: metadata field '{field}' "
                            "is not valid for a zone"
                        )
                    capacity = values.get("max_drones", "1")
                    if not re.fullmatch(r"[0-9]+", capacity):
                        raise ConfigParseError(
                            f"line {line_number}: max_drones must be a "
                            "positive integer"
                        )
                    try:
                        zone_values["max_drones"] = int(capacity)
                    except ValueError as error:
                        raise ConfigParseError(
                            f"line {line_number}: max_drones integer "
                            "exceeds the supported numeric representation"
                        ) from error
                    regular_zone = self._validate_model(
                        RegZoneSpec, zone_values, line_number, "zone"
                    )
                    zones.append(regular_zone)

                defined_names.add(zone_name)
            except ConfigParseError:
                raise

        if start is None:
            raise ConfigParseError(
                f"after line {records[-1][0]}: missing required "
                "start_hub definition"
            )
        if end is None:
            raise ConfigParseError(
                f"after line {records[-1][0]}: missing required "
                "end_hub definition"
            )

        try:
            return MapConfig(
                nb_drones=nb_drones,
                start=start,
                end=end,
                zones=zones,
                connections=connections,
            )
        except ValidationError as error:
            detail = error.errors()[0]
            location = ".".join(str(part) for part in detail["loc"])
            raise ConfigParseError(
                f"invalid map configuration at {location}: {detail['msg']}"
            ) from error

    @staticmethod
    def _split_metadata(
        body: str, line_number: int, primary_fields: int
    ) -> tuple[str, str | None]:
        """Separate trailing metadata without consuming brackets in names."""
        parts = body.split(maxsplit=primary_fields)
        if len(parts) <= primary_fields:
            return body, None
        primary = " ".join(parts[:primary_fields])
        match = re.fullmatch(r"\[([^\[\]]*)\]\s*", parts[-1])
        if not match:
            raise ConfigParseError(
                f"line {line_number}: metadata must be one bracketed block "
                "at the end of the line"
            )
        return primary, match.group(1)

    @staticmethod
    def _parse_metadata(raw: str | None, line_number: int) -> dict[str, str]:
        """Validate known, unique key=value metadata entries."""
        if raw is None:
            return {}
        values: dict[str, str] = {}
        for item in raw.split():
            key, separator, value = item.partition("=")
            if not separator or not key or not value:
                raise ConfigParseError(
                    f"line {line_number}: invalid metadata item '{item}', "
                    "expected key=value"
                )
            if key not in {"zone", "color", "max_drones", "max_link_capacity"}:
                raise ConfigParseError(
                    f"line {line_number}: unknown metadata field '{key}'"
                )
            if key in values:
                raise ConfigParseError(
                    f"line {line_number}: metadata field '{key}' is "
                    "specified more than once"
                )
            values[key] = value
        return values

    @staticmethod
    def _parse_zone_coordinates(
        primary: str, line_number: int
    ) -> tuple[str, int, int]:
        """Parse a hub name and two exact integer coordinates."""
        parts = primary.split()
        if len(parts) != 3:
            raise ConfigParseError(
                f"line {line_number}: zone definition must contain a name "
                "and two integer coordinates"
            )
        try:
            if not re.fullmatch(r"[+-]?\d+", parts[1]) or not re.fullmatch(
                r"[+-]?\d+", parts[2]
            ):
                raise ValueError
            x, y = int(parts[1]), int(parts[2])
        except ValueError as error:
            raise ConfigParseError(
                f"line {line_number}: zone coordinates must be integers"
            ) from error
        if "-" in parts[0]:
            raise ConfigParseError(
                f"line {line_number}: zone names cannot contain dashes"
            )
        return parts[0], x, y

    @classmethod
    def _parse_connection(
        cls,
        primary: str,
        values: dict[str, str],
        line_number: int,
        defined_names: set[str],
    ) -> RouteSpec:
        """Validate a link against previously declared endpoints."""
        if set(values) - {"max_link_capacity"}:
            invalid = sorted(set(values) - {"max_link_capacity"})[0]
            raise ConfigParseError(
                f"line {line_number}: metadata field '{invalid}' is not "
                "valid for a connection"
            )
        names = primary.split("-")
        if len(names) != 2 or not all(names):
            raise ConfigParseError(
                f"line {line_number}: connection must use '<zone1>-<zone2>'"
            )
        from_hub, to_hub = names
        for name in names:
            if name not in defined_names:
                raise ConfigParseError(
                    f"line {line_number}: connection references undefined "
                    f"or not-yet-defined zone '{name}'"
                )
        if from_hub == to_hub:
            raise ConfigParseError(
                f"line {line_number}: a connection must join two "
                "different zones"
            )

        capacity = values.get("max_link_capacity", "1")
        try:
            if not re.fullmatch(r"[0-9]+", capacity):
                raise ValueError
            capacity_value = int(capacity)
        except ValueError as error:
            raise ConfigParseError(
                f"line {line_number}: max_link_capacity must be a "
                "positive integer"
            ) from error
        return cls._validate_model(
            RouteSpec,
            {
                "from_hub": from_hub,
                "to_hub": to_hub,
                "max_link_capacity": capacity_value,
            },
            line_number,
            "connection",
        )

    @staticmethod
    def _validate_model(
        model: type[ModelT],
        values: dict[str, Any],
        line_number: int,
        kind: str,
    ) -> ModelT:
        """Convert model errors into a line-specific configuration error."""
        try:
            return model.model_validate(values)
        except ValidationError as error:
            detail = error.errors()[0]
            location = ".".join(str(part) for part in detail["loc"])
            raise ConfigParseError(
                f"line {line_number}: invalid {kind} {location}: "
                f"{detail['msg']}"
            ) from error
