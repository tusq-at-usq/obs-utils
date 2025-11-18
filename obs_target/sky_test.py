from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from numpy.typing import ArrayLike, NDArray
import numpy as np

from obs_target.target import Target

import astrix as at
from astrix.spatial import Rotation

BRIGHT_STARS = {
    "Sirius": ("06h45m08.917s", "-16d42m58.02s"),
    "Canopus": ("06h23m57.109s", "-52d41m44.38s"),
    "Arcturus": ("14h15m39.672s", "+19d10m56.67s"),
    "Vega": ("18h36m56.336s", "+38d47m01.28s"),
    "Capella": ("05h16m41.358s", "+45d59m52.77s"),
    "Rigel": ("05h14m32.272s", "-08d12m05.90s"),
    "Procyon": ("07h39m18.119s", "+05d13m29.96s"),
    "Achernar": ("01h37m42.845s", "-57d14m12.33s"),
    "Betelgeuse": ("05h55m10.305s", "+07d24m25.43s"),
    "Hadar": ("14h03m49.405s", "-60d22m22.72s"),
    "Altair": ("19h50m47.005s", "+08d52m05.96s"),
    "Aldebaran": ("04h35m55.239s", "+16d30m33.49s"),
    "Antares": ("16h29m24.459s", "-26d25m55.21s"),
    "Spica": ("13h25m11.579s", "-11d09m40.75s"),
    "Fomalhaut": ("22h57m39.046s", "-29d37m20.05s"),
}
PLANETS = [
    "earth",
    "sun",
    "moon",
    "mercury",
    "venus",
    "earth-moon-barycenter",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]


class SkyTarget(Target):
    _name: str
    _point: at.Point

    def __init__(self, name: str, point: at.Point) -> None:
        self._name = name
        self._point = point
        self._frame_ned = at.spatial.frame.ned_frame(self._point)

    def check_time_bounds(self, t_unix: float) -> tuple[bool, bool]:
        return True, True

    def get_planet(self, name: str, t_unix: float) -> tuple[float, float]:
        lat, lon, height = self._point.geodet[0]
        t = Time(t_unix, format="unix")
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.meter)

        with solar_system_ephemeris.set("jpl"):
            planet = get_body(name, t, loc)

        altazframe = AltAz(obstime=t, location=loc, pressure=0)
        planetaz = planet.transform_to(altazframe)
        az = planetaz.az.degree
        el = planetaz.alt.degree
        return az, el

    def get_star(self, name: str, t_unix: float) -> tuple[float, float]:
        lat, lon, height = self._point.geodet[0]
        time = Time(t_unix, format="unix")
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.meter)
        ra, dec = BRIGHT_STARS[name]
        star_icrs = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="icrs")
        altaz = star_icrs.transform_to(AltAz(obstime=time, location=loc))
        az = altaz.az.degree
        el = altaz.alt.degree
        return az, el

    def _get_az_el(self, name: str, t_unix: float) -> tuple[float, float]:
        """Get azimuth and elevation of a sky target (star or planet) from a given time and location.
        Args:
            name (str): Name of the sky target (star or planet).
            t_unix (float): Unix time to get the azimuth and elevation.
        Returns:
            az (float): Azimuth in degrees.
            el (float): Elevation in degrees.
        """  # Get current location from target path

        if name in BRIGHT_STARS:
            return self.get_star(name, t_unix)
        elif name in PLANETS:
            return self.get_planet(name, t_unix)
        else:
            raise ValueError(f"Unknown sky target name: {name}")

    def get_head_pitch(self, t_unix: float) -> NDArray:
        az, el = self._get_az_el(self._name, t_unix)
        heading = az
        pitch = el
        return np.array([heading, pitch])

    def get_head_pitch_rate(self, t_unix: float) -> NDArray:
        return np.array([0.0, 0.0])

    def project_from_ned_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[NDArray, NDArray]:
        hp = self.get_head_pitch(t_unix)
        rot = Rotation.from_euler("ZYX", np.array(euler).reshape(1, -1), degrees=True)
        frame = at.Frame(rot, ref_frame=self._frame_ned)
        ray = at.Ray.from_az_el(
            az_el=hp,
            frame=frame,
            time=at.Time(t_unix),
        )
        uv = ray.project_to_cam(cam)
        return uv.uv, uv.uv

    def project_from_ecef_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[NDArray, NDArray]:
        hp = self.get_head_pitch(t_unix)
        rot = Rotation.from_euler("ZYX", np.array(euler).reshape(1, -1), degrees=True)
        frame = at.Frame(rot)
        ray = at.Ray.from_az_el(
            az_el=hp,
            frame=frame,
            time=at.Time(t_unix),
        )
        uv = ray.project_to_cam(cam)
        return uv.uv, uv.uv
