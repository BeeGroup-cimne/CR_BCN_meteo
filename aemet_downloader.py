from aemet_opendata.interface import ConnectionOptions, UpdateFeature
import json
from typing import Any, Final, cast
import asyncio
import timeit
import aiohttp
import os
import logging
import base64
from datetime import timedelta, datetime, timezone, UTC
from aemet_opendata.interface import AEMET
from aemet_opendata.const import (
    API_CALL_DATA_TIMEOUT_DEF,
    API_CALL_FILE_EXTENSION,
    ATTR_BYTES,
    ATTR_DATA,
    ATTR_TIMESTAMP
)
from aemet_opendata.exceptions import AuthError
from aemet_opendata.helpers import (
    BytesEncoder,
    get_current_datetime,
    parse_api_timestamp,
    slugify,
)


# AEMET_COORDS = [(41.293,2.07), (41.418, 2.124), (41.533, 2.037), (41.608, 2.298)]
# AEMET_STATIONS = ["BARCELONA/AEROPUERTO", "BARCELONA (FABRA)", "TERRASSA (LES FONTS)", "GRANOLLERS"]
# AEMET_ID = ["0076", "0200E", "0189E", "0208"]

AEMET_DATA_DIR = "/home/gmor/Nextcloud2/Beegroup/data/AEMET"
os.makedirs(AEMET_DATA_DIR, exist_ok=True)
AEMET_OPTIONS = ConnectionOptions(
    api_key="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJnbW9yQGNpbW5lLnVwYy5lZHUiLCJqdGkiOiI0YjAyYTFiYi1kMjQ5LTQ3OWUtODk5ZC0zOWQzMWY4MzBjMmMiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc0NzM4MjYzMiwidXNlcklkIjoiNGIwMmExYmItZDI0OS00NzllLTg5OWQtMzlkMzFmODMwYzJjIiwicm9sZSI6IiJ9.Xzy1t_rjGShushlBBUJ9FnSOMbncuqP06rEnyCAgNjE",
    update_features=UpdateFeature.STATION | UpdateFeature.RADAR,
)

def json_dumps(data: Any) -> str:
    """Dump data using json.dumps()."""
    return json.dumps(data, indent=4, sort_keys=True, default=str)

class AEMET_v2(AEMET):
    _LOGGER = logging.getLogger(__name__)

    API_CALL_DATA_TIMEOUT: Final[dict[str, timedelta]] = {
        "maestro/municipios": timedelta(days=15),
        "prediccion/especifica/municipio/diaria": timedelta(days=3),
        "prediccion/especifica/municipio/horaria": timedelta(hours=48),
        "observacion/convencional/datos/estacion": timedelta(hours=2),
        "observacion/convencional/todas": timedelta(days=15),
        "red/radar/nacional": timedelta(hours=6),
        "red/rayos/mapa": timedelta(hours=6),
    }
    async def get_conventional_observation_all_stations_data(
        self, fetch_data: bool = True,
    ) -> dict[str, Any]:
        endpoint = (
            f"observacion/convencional/todas"
        )
        return await self.api_call(endpoint, fetch_data)

    async def api_call_save(self, cmd: str, json_data: dict[str, Any]) -> None:
        """Save API call to file."""
        if self._api_data_dir is None:
            return

        file_name = f"{slugify(cmd)}_{datetime.now(UTC).isoformat()}{API_CALL_FILE_EXTENSION}"
        file_path = os.path.join(self._api_data_dir, file_name)
        file_data = json.dumps(json_data, cls=BytesEncoder)
        await self.api_file_write(file_path, file_data)

    async def api_call_load(self, cmd: str) -> dict[str, Any] | None:
        """Load API call from file."""
        json_data: dict[str, Any] | None = None

        if self._api_data_dir is None:
            return None

        file_name = f"{slugify(cmd)}_{datetime.now(UTC).isoformat()}{API_CALL_FILE_EXTENSION}"
        file_path = os.path.join(self._api_data_dir, file_name)
        if not os.path.isfile(file_path):
            return None

        data_timeout = API_CALL_DATA_TIMEOUT_DEF
        for key, val in self.API_CALL_DATA_TIMEOUT.items():
            if cmd.startswith(key):
                data_timeout = val
                break

        self._LOGGER.info('Loading cmd=%s from "%s"...', cmd, file_name)

        file_bytes = await self.api_file_read(file_path)
        json_data = json.loads(file_bytes)

        json_data = cast(dict[str, Any], json_data)

        file_isotime = json_data.get(ATTR_TIMESTAMP)
        if file_isotime is not None:
            file_datetime = parse_api_timestamp(file_isotime)
        else:
            file_mtime = os.path.getmtime(file_path)
            file_datetime = datetime.fromtimestamp(file_mtime, tz=timezone.utc)

        cur_datetime = get_current_datetime(replace=False)
        if cur_datetime - file_datetime > data_timeout:
            return None

        json_attr_data = json_data.get(ATTR_DATA, {})
        if isinstance(json_attr_data, dict):
            json_bytes = json_attr_data.get(ATTR_BYTES)
            if json_bytes is not None:
                json_data[ATTR_DATA][ATTR_BYTES] = base64.b64decode(json_bytes)

        return json_data

async def main():
    """AEMET OpenData client example."""

    async with aiohttp.ClientSession() as aiohttp_session:
        client = AEMET_v2(aiohttp_session, AEMET_OPTIONS)

        client.set_api_data_dir(AEMET_DATA_DIR)

        try:
            get_town_start = timeit.default_timer()
            town = await client.get_conventional_observation_all_stations_data()
            get_town_end = timeit.default_timer()
            # print(json_dumps(town))
            print(f"Get Town time: {get_town_end - get_town_start}")
        except AuthError as e:
            print(f"API authentication error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
