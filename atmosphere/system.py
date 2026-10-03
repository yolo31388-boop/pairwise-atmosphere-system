"""
pairwise-atmosphere-system - AtmosphereSystem

Physical atmosphere model:

* Solar motion driven by latitude, day of year (solar declination) and time,
  with computed sunrise/sunset times and smooth interpolation.
* Sky colour from atmospheric scattering (Rayleigh + Mie + multiple
  scattering) with a sun halo and full day/night cycle.
* Animated 3-D volumetric clouds with wind advection and directional
  lighting (bright tops, dark bottoms, self shadowing).
* Smooth weather transitions over cloud cover, light and fog with continuous
  intermediate states between named weather presets.

The module only depends on the Python standard library.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 == edge0:
        return 1.0 if value >= edge1 else 0.0
    t = _clamp((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _vec_mix(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (_mix(a[0], b[0], t), _mix(a[1], b[1], t), _mix(a[2], b[2], t))


def _normalize(v: Vec3) -> Vec3:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / length
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class _ValueNoise3D:
    """Deterministic trilinearly interpolated value noise in 3-D."""

    def __init__(self, seed: int = 1337):
        rand = random.Random(seed)
        size = 256
        self.size = size
        self.values = [rand.uniform(-1.0, 1.0) for _ in range(size)]
        permutation = list(range(size))
        rand.shuffle(permutation)
        self.perm = permutation + permutation

    def _lattice(self, ix: int, iy: int, iz: int) -> float:
        p = self.perm
        h = p[(p[(p[ix & 255] + iy) & 255] + iz) & 255]
        return self.values[h]

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def sample(self, x: float, y: float, z: float) -> float:
        x0, y0, z0 = math.floor(x), math.floor(y), math.floor(z)
        fx, fy, fz = x - x0, y - y0, z - z0
        ux, uy, uz = self._fade(fx), self._fade(fy), self._fade(fz)
        c000 = self._lattice(x0, y0, z0)
        c100 = self._lattice(x0 + 1, y0, z0)
        c010 = self._lattice(x0, y0 + 1, z0)
        c110 = self._lattice(x0 + 1, y0 + 1, z0)
        c001 = self._lattice(x0, y0, z0 + 1)
        c101 = self._lattice(x0 + 1, y0, z0 + 1)
        c011 = self._lattice(x0, y0 + 1, z0 + 1)
        c111 = self._lattice(x0 + 1, y0 + 1, z0 + 1)
        x00 = _mix(c000, c100, ux)
        x10 = _mix(c010, c110, ux)
        x01 = _mix(c001, c101, ux)
        x11 = _mix(c011, c111, ux)
        xy0 = _mix(x00, x10, uy)
        xy1 = _mix(x01, x11, uy)
        return _mix(xy0, xy1, uz)

    def fbm(self, x: float, y: float, z: float, octaves: int = 4) -> float:
        total = 0.0
        amplitude = 0.5
        frequency = 1.0
        normalization = 0.0
        for _ in range(octaves):
            total += amplitude * self.sample(x * frequency, y * frequency, z * frequency)
            normalization += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        return total / normalization if normalization else 0.0


WEATHER_PRESETS: Dict[str, Dict[str, float]] = {
    "clear": {
        "cloud_cover": 0.05,
        "cloud_density": 0.35,
        "light_multiplier": 1.00,
        "sky_darkening": 0.00,
        "turbidity": 2.5,
        "fog_density": 0.00,
        "precipitation": 0.0,
    },
    "partly_cloudy": {
        "cloud_cover": 0.35,
        "cloud_density": 0.50,
        "light_multiplier": 0.90,
        "sky_darkening": 0.08,
        "turbidity": 3.2,
        "fog_density": 0.02,
        "precipitation": 0.0,
    },
    "cloudy": {
        "cloud_cover": 0.65,
        "cloud_density": 0.70,
        "light_multiplier": 0.70,
        "sky_darkening": 0.22,
        "turbidity": 4.5,
        "fog_density": 0.06,
        "precipitation": 0.0,
    },
    "overcast": {
        "cloud_cover": 0.90,
        "cloud_density": 0.90,
        "light_multiplier": 0.50,
        "sky_darkening": 0.40,
        "turbidity": 6.0,
        "fog_density": 0.12,
        "precipitation": 0.1,
    },
    "rainy": {
        "cloud_cover": 1.00,
        "cloud_density": 1.00,
        "light_multiplier": 0.35,
        "sky_darkening": 0.55,
        "turbidity": 8.0,
        "fog_density": 0.20,
        "precipitation": 0.8,
    },
}

WEATHER_PARAMETERS = tuple(WEATHER_PRESETS["clear"].keys())


class AtmosphereSystem:
    """Physical sun/sky/cloud/weather simulation."""

    EARTH_AXIAL_TILT = 23.44

    def __init__(
        self,
        latitude: float = 45.0,
        day_of_year: int = 172,
        time_of_day: float = 12.0,
        weather: str = "clear",
        seed: int = 1337,
    ):
        self.latitude = float(latitude)
        self.day_of_year = int(day_of_year)
        self.time_of_day = float(time_of_day)
        self.weather_name = weather
        self.cloud_time = 0.0
        self.simulation_time = 0.0
        self.wind: Vec3 = (1.5, 0.4, 0.0)

        self.noise = _ValueNoise3D(seed)
        self.cloud_grid_shape: Tuple[int, int, int] = (16, 16, 8)
        self.cloud_density_grid: list[float] = [0.0] * (
            self.cloud_grid_shape[0]
            * self.cloud_grid_shape[1]
            * self.cloud_grid_shape[2]
        )
        self.cloud_light_grid: list[float] = [0.0] * len(self.cloud_density_grid)

        # Precomputed scattering lookup table, keyed by (sun elevation bin,
        # turbidity bin, view elevation bin). Filled lazily by
        # ``calculate_scattering``.
        self._scattering_lut: Dict[Tuple[int, int, int], Dict[str, object]] = {}
        self._scattering_lut_bins = 96

        sunrise, sunset = self.compute_sunrise_sunset(self.latitude, self.day_of_year)
        self.sunrise = sunrise
        self.sunset = sunset
        sun = self._compute_solar_coordinates(
            self.time_of_day, self.latitude, self.day_of_year
        )
        self.sun_azimuth = sun["azimuth"]
        self.sun_elevation = sun["elevation"]
        self.sun_direction: Vec3 = sun["direction"]
        self._last_sun: Optional[Dict[str, float]] = None

        self.weather = dict(WEATHER_PRESETS[weather])
        self.weather_target_name = weather
        self.weather_target = dict(self.weather)
        self._transition_start: Optional[float] = None
        self._transition_duration = 0.0
        self._transition_from = dict(self.weather)

        self.state = {
            "latitude": self.latitude,
            "day_of_year": self.day_of_year,
            "time_of_day": self.time_of_day,
            "sun_azimuth": self.sun_azimuth,
            "sun_elevation": self.sun_elevation,
            "weather": self.weather,
        }

    # ------------------------------------------------------------------
    # Solar motion
    # ------------------------------------------------------------------
    @classmethod
    def solar_declination(cls, day_of_year: int) -> float:
        """Solar declination in radians for a day of year (1..365)."""
        angle = math.radians(360.0 * (float(day_of_year) + 10.0) / 365.25)
        return -math.radians(cls.EARTH_AXIAL_TILT) * math.cos(angle)

    @staticmethod
    def compute_sunrise_sunset(
        latitude: float, day_of_year: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """Local solar sunrise/sunset hours, accounting for polar day/night."""
        declination = AtmosphereSystem.solar_declination(day_of_year)
        phi = math.radians(latitude)
        cos_hour_angle = -math.tan(phi) * math.tan(declination)
        if cos_hour_angle <= -1.0:
            return 0.0, 24.0  # polar day / midnight sun
        if cos_hour_angle >= 1.0:
            return None, None  # polar night
        half_day = math.degrees(math.acos(cos_hour_angle)) / 15.0
        sunrise = 12.0 - half_day
        sunset = 12.0 + half_day
        return sunrise, sunset

    def _compute_solar_coordinates(
        self, time_of_day: float, latitude: float, day_of_year: int
    ) -> Dict[str, object]:
        declination = self.solar_declination(day_of_year)
        phi = math.radians(latitude)
        hour_angle = math.radians(15.0 * (time_of_day - 12.0))

        sin_elevation = (
            math.sin(phi) * math.sin(declination)
            + math.cos(phi) * math.cos(declination) * math.cos(hour_angle)
        )
        elevation = math.asin(_clamp(sin_elevation, -1.0, 1.0))

        cos_azimuth = (
            math.sin(declination) - math.sin(elevation) * math.sin(phi)
        ) / (math.cos(elevation) * math.cos(phi) + 1e-9)
        cos_azimuth = _clamp(cos_azimuth, -1.0, 1.0)
        azimuth = math.acos(cos_azimuth)
        if math.sin(hour_angle) > 0.0:
            azimuth = 2.0 * math.pi - azimuth

        cos_elev = math.cos(elevation)
        # x east, y north, z up
        direction = (
            -cos_elev * math.sin(azimuth),
            cos_elev * math.cos(azimuth),
            math.sin(elevation),
        )
        sunrise, sunset = self.compute_sunrise_sunset(latitude, day_of_year)
        if sunrise is not None and sunset is not None:
            day_length = sunset - sunrise
        elif sin_elevation > 0.0:
            day_length = 24.0
        else:
            day_length = 0.0
        return {
            "elevation": elevation,
            "azimuth": azimuth,
            "direction": direction,
            "declination": declination,
            "sunrise": sunrise,
            "sunset": sunset,
            "day_length": day_length,
            "is_polar_day": (sunrise == 0.0 and sunset == 24.0),
            "is_polar_night": (sunrise is None),
        }

    def update_sun_position(
        self,
        time_of_day: Optional[float] = None,
        latitude: Optional[float] = None,
        day_of_year: Optional[int] = None,
        dt: float = 0.0,
        smoothing: float = 0.35,
    ) -> Dict[str, object]:
        """Update the sun position for the given time/location.

        * Latitude and day of year drive the declination, so equator and
          polar trajectories differ correctly.
        * Sunrise/sunset are recomputed for the new location/season.
        * ``dt`` advances the simulation clock (seconds); positions are
          smoothed towards the exact solution so the sun never jumps.
        """
        if latitude is not None:
            self.latitude = float(latitude)
        if day_of_year is not None:
            self.day_of_year = int(day_of_year)
        if time_of_day is not None:
            self.time_of_day = float(time_of_day) % 24.0
        elif dt:
            self.simulation_time += dt
            self.time_of_day = (self.time_of_day + dt / 3600.0) % 24.0

        exact = self._compute_solar_coordinates(
            self.time_of_day, self.latitude, self.day_of_year
        )
        explicit_set = (
            time_of_day is not None
            or latitude is not None
            or day_of_year is not None
        )

        if self._last_sun is None or explicit_set or smoothing >= 1.0:
            smoothed_elevation = exact["elevation"]
            smoothed_azimuth = exact["azimuth"]
        else:
            blend = _clamp(smoothing)
            smoothed_elevation = _mix(
                self._last_sun["elevation"], exact["elevation"], blend
            )
            prev_az = self._last_sun["azimuth"]
            delta = (exact["azimuth"] - prev_az + math.pi) % (2.0 * math.pi) - math.pi
            smoothed_azimuth = (prev_az + blend * delta) % (2.0 * math.pi)

        self._last_sun = {
            "elevation": smoothed_elevation,
            "azimuth": smoothed_azimuth,
        }
        self.sun_elevation = smoothed_elevation
        self.sun_azimuth = smoothed_azimuth
        cos_elev = math.cos(smoothed_elevation)
        self.sun_direction = (
            -cos_elev * math.sin(smoothed_azimuth),
            cos_elev * math.cos(smoothed_azimuth),
            math.sin(smoothed_elevation),
        )
        self.sunrise = exact["sunrise"]
        self.sunset = exact["sunset"]

        result = {
            "azimuth": smoothed_azimuth,
            "elevation": smoothed_elevation,
            "direction": self.sun_direction,
            "declination": exact["declination"],
            "sunrise": exact["sunrise"],
            "sunset": exact["sunset"],
            "day_length": exact["day_length"],
            "is_day": smoothed_elevation > 0.0,
            "is_polar_day": exact["is_polar_day"],
            "is_polar_night": exact["is_polar_night"],
            "time_of_day": self.time_of_day,
            "latitude": self.latitude,
            "day_of_year": self.day_of_year,
        }
        self.state.update(
            {
                "sun_azimuth": smoothed_azimuth,
                "sun_elevation": smoothed_elevation,
                "time_of_day": self.time_of_day,
                "latitude": self.latitude,
                "day_of_year": self.day_of_year,
            }
        )
        return result

    # ------------------------------------------------------------------
    # Atmospheric scattering
    # ------------------------------------------------------------------
    RAYLEIGH_BETA: Vec3 = (5.5e-6, 13.0e-6, 22.4e-6)  # 1/m at sea level
    MIE_BETA: float = 21e-6                            # 1/m at sea level
    RAYLEIGH_SCALE_HEIGHT = 7994.0                     # m
    MIE_SCALE_HEIGHT = 1200.0                          # m
    MIE_ASYMMETRY = 0.76

    @staticmethod
    def _optical_depth(scale_height: float, elevation: float) -> float:
        """Approximate air mass along a ray at the given elevation."""
        sin_elev = max(math.sin(max(elevation, 0.0)), 0.0)
        # Kasten-Young style air mass approximation.
        zenith = math.degrees(math.acos(_clamp(sin_elev)))
        air_mass = 1.0 / (
            math.cos(math.radians(zenith))
            + 0.50572 * (96.07995 - zenith) ** -1.6364
        )
        return air_mass

    def _scattering_components(
        self, sun_elevation: float, view_elevation: float, turbidity: float
    ) -> Dict[str, object]:
        """Full Rayleigh + Mie (+ multiple scattering) sky radiance model."""
        air_mass_sun = self._optical_depth(
            self.RAYLEIGH_SCALE_HEIGHT, sun_elevation
        )
        air_mass_view = self._optical_depth(
            self.RAYLEIGH_SCALE_HEIGHT, view_elevation
        )
        mie_beta = self.MIE_BETA * turbidity
        od_sun = tuple(
            (self.RAYLEIGH_BETA[i] * self.RAYLEIGH_SCALE_HEIGHT
             + mie_beta * self.MIE_SCALE_HEIGHT) * air_mass_sun
            for i in range(3)
        )
        od_view = tuple(
            (self.RAYLEIGH_BETA[i] * self.RAYLEIGH_SCALE_HEIGHT
             + mie_beta * self.MIE_SCALE_HEIGHT) * air_mass_view
            for i in range(3)
        )

        # Transmittance of the direct sun beam: blue is extinguished first,
        # which is what reddens sunlight near the horizon.
        transmittance = tuple(math.exp(-od_sun[i]) for i in range(3))

        # Single-scatter in-scatter along the view path, lit by the
        # (already attenuated) sun beam.
        rayleigh_phase = 0.75  # ~ (3/4)(1+cos^2) averaged over the dome
        mie_phase_avg = 0.2
        inscatter = []
        for i in range(3):
            r_coeff = self.RAYLEIGH_BETA[i] * self.RAYLEIGH_SCALE_HEIGHT
            m_coeff = mie_beta * self.MIE_SCALE_HEIGHT
            total = r_coeff + m_coeff
            weight = (r_coeff * rayleigh_phase + m_coeff * mie_phase_avg) / total
            inscatter.append(
                transmittance[i] * (1.0 - math.exp(-od_view[i])) * weight
            )
        inscatter = tuple(inscatter)

        # Multiple scattering: energy scattered once is scattered again,
        # brightening and desaturating the sky. Approximate with a geometric
        # series capped at second order.
        multi_strength = tuple(
            0.5 * od_view[i] / (1.0 + od_view[i]) for i in range(3)
        )
        radiance = tuple(
            inscatter[i] * (1.0 + multi_strength[i] + multi_strength[i] ** 2)
            for i in range(3)
        )
        return {
            "rayleigh_optical_depth": tuple(
                self.RAYLEIGH_BETA[i] * self.RAYLEIGH_SCALE_HEIGHT * air_mass_view
                for i in range(3)
            ),
            "mie_optical_depth": mie_beta * self.MIE_SCALE_HEIGHT * air_mass_view,
            "transmittance": transmittance,
            "radiance": radiance,
            "air_mass": 0.5 * (air_mass_sun + air_mass_view),
            "multiple_scattering": multi_strength,
        }

    def calculate_scattering(
        self,
        sun_elevation: Optional[float] = None,
        view_elevation: float = math.radians(45.0),
        turbidity: Optional[float] = None,
        use_lut: bool = True,
    ) -> Dict[str, object]:
        """Rayleigh + Mie + multiple-scattering solution.

        Results are cached in a precomputed lookup table keyed by binned
        sun elevation / turbidity / view elevation, so per-frame calls are
        cheap table lookups instead of full recomputation.
        """
        if sun_elevation is None:
            sun_elevation = self.sun_elevation
        if turbidity is None:
            turbidity = self.weather["turbidity"]

        bins = self._scattering_lut_bins
        key = (
            int(round(math.degrees(sun_elevation) / 180.0 * bins)),
            int(round(turbidity * 4.0)),
            int(round(math.degrees(view_elevation) / 90.0 * bins)),
        )
        if use_lut and key in self._scattering_lut:
            return self._scattering_lut[key]

        result = self._scattering_components(sun_elevation, view_elevation, turbidity)
        result["rayleigh_beta"] = self.RAYLEIGH_BETA
        result["mie_beta"] = self.MIE_BETA * turbidity
        result["turbidity"] = turbidity
        if use_lut:
            self._scattering_lut[key] = result
        return result

    # ------------------------------------------------------------------
    # Sky colour
    # ------------------------------------------------------------------
    @staticmethod
    def _mie_phase(cos_theta: float, g: float) -> float:
        g2 = g * g
        numerator = (1.0 - g2) * (1.0 + cos_theta * cos_theta)
        denominator = (2.0 + g2) * (
            1.0 + g2 - 2.0 * g * cos_theta
        ) ** 1.5
        return 0.119366 * numerator / max(denominator, 1e-9)

    def calculate_sky_color(
        self,
        view_direction: Optional[Vec3] = None,
        view_elevation: float = math.radians(45.0),
        view_azimuth: float = 0.0,
        sun_elevation: Optional[float] = None,
        sun_azimuth: Optional[float] = None,
        turbidity: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Scattering-based sky colour with sun halo and day/night cycle."""
        if sun_elevation is None:
            sun_elevation = self.sun_elevation
        if sun_azimuth is None:
            sun_azimuth = self.sun_azimuth
        if turbidity is None:
            turbidity = self.weather["turbidity"]
        if view_direction is None:
            view_direction = (
                -math.cos(view_elevation) * math.sin(view_azimuth),
                math.cos(view_elevation) * math.cos(view_azimuth),
                math.sin(view_elevation),
            )
        view_direction = _normalize(view_direction)

        cos_angle = _clamp(_dot(view_direction, self.sun_direction), -1.0, 1.0)

        scatter = self.calculate_scattering(
            sun_elevation=sun_elevation,
            view_elevation=view_elevation,
            turbidity=turbidity,
        )
        radiance = scatter["radiance"]
        transmittance = scatter["transmittance"]

        # Warm sunlight colour at low sun (extra red/red-orange attenuation).
        sun_height = _clamp(math.sin(sun_elevation))
        sunlight = (
            1.0 - 0.75 * (1.0 - transmittance[0]),
            0.5 + 0.5 * transmittance[1],
            0.25 + 0.75 * transmittance[2],
        )

        # Mie forward scattering creates the halo/glow around the sun; the
        # Henyey-Greenstein phase is strongly forward peaked, so the glow
        # only appears when looking toward the sun.
        horizon_boost = 0.6 + 1.8 * _smoothstep(-0.1, 0.25, sun_elevation)
        halo = _clamp(
            self._mie_phase(cos_angle, self.MIE_ASYMMETRY) * 0.35 * horizon_boost
        )

        # Day weight: full daylight above ~6 deg, twilight band, night below.
        day = _smoothstep(math.radians(-6.0), math.radians(8.0), sun_elevation)
        twilight = math.exp(
            -((sun_elevation - math.radians(-2.0)) ** 2)
            / (2.0 * math.radians(5.0) ** 2)
        )

        sky: Vec3 = tuple(
            radiance[i] * _mix(sunlight[i], 1.0, 0.4)
            for i in range(3)
        )
        sky = _vec_mix(sky, sunlight, halo)

        # Twilight warm band near the horizon.
        twilight_color = (0.85, 0.35, 0.12)
        sky = tuple(
            sky[i] * day
            + twilight_color[i] * twilight * (1.0 - day)
            + 0.01 * (0.4, 0.5, 0.9)[i] * (1.0 - day) * (1.0 - twilight)
            for i in range(3)
        )

        # Weather darkening from cloud cover / overcast conditions.
        darkening = self.weather["sky_darkening"]
        sky = tuple(c * (1.0 - darkening * (0.4 + 0.6 * day)) for c in sky)

        # Exponential tone map into display range.
        return tuple(1.0 - math.exp(-c * 8.0) for c in sky)

    # ------------------------------------------------------------------
    # 3-D volumetric clouds
    # ------------------------------------------------------------------
    def _cloud_index(self, x: int, y: int, z: int) -> int:
        nx, ny, _ = self.cloud_grid_shape
        return (z * ny + y) * nx + x

    def simulate_clouds(
        self,
        dt: float = 0.0,
        coverage: Optional[float] = None,
        density: Optional[float] = None,
        wind: Optional[Sequence[float]] = None,
        time: Optional[float] = None,
    ) -> Dict[str, object]:
        """Advect and light a 3-D volumetric cloud field.

        * Density comes from 3-D fBm value noise sampled at a wind-offset
          position, so clouds have real thickness and move over time.
        * Coverage is carved out by remapping the density field.
        * Lighting marches toward the sun per column: cloud tops catch
          direct light, bases stay dark, and dense cloud above casts
          self-shadows.
        """
        if wind is not None:
            self.wind = (float(wind[0]), float(wind[1]), float(wind[2]))
        if time is not None:
            self.cloud_time = float(time)
        elif dt:
            self.cloud_time += dt

        coverage = self.weather["cloud_cover"] if coverage is None else coverage
        density = self.weather["cloud_density"] if density is None else density

        nx, ny, nz = self.cloud_grid_shape
        speed = math.hypot(self.wind[0], self.wind[1])
        offset_x = self.wind[0] * self.cloud_time * 0.02
        offset_y = self.wind[1] * self.cloud_time * 0.02
        scale = 0.35

        threshold = 1.0 - _clamp(coverage)

        # Density field (3-D, advected by the wind).
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    sample = 0.5 + 0.5 * self.noise.fbm(
                        x * scale + offset_x,
                        y * scale + offset_y,
                        z * scale * 0.8,
                    )
                    # Vertical profile: thicker around mid height, fading out.
                    vertical = 1.0 - abs(z / max(nz - 1, 1) * 2.0 - 1.0)
                    sample *= 0.55 + 0.45 * vertical
                    cloud = _clamp(
                        (sample - threshold) / max(1.0 - threshold, 1e-3)
                    )
                    self.cloud_density_grid[self._cloud_index(x, y, z)] = (
                        cloud * density
                    )

        # Directional lighting from the sun (with shadows and dark bases).
        sun_visible = _clamp(math.sin(self.sun_elevation) * 1.4)
        direct_light = 1.25 * sun_visible * self.weather["light_multiplier"]
        ambient = 0.18 + 0.12 * _clamp(math.sin(self.sun_elevation) + 0.2)

        sun_dx, sun_dy, sun_dz = _normalize(self.sun_direction)
        light_steps = 4
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    idx = self._cloud_index(x, y, z)
                    d = self.cloud_density_grid[idx]
                    if d <= 0.001:
                        self.cloud_light_grid[idx] = ambient * 0.5
                        continue

                    # March upward toward the sun to find extinction.
                    optical = 0.0
                    for step in range(1, light_steps + 1):
                        sx = (x + sun_dx * step * 0.7) % nx
                        sy = (y + sun_dy * step * 0.7) % ny
                        sz = _clamp(z + sun_dz * step * 0.7, 0, nz - 1)
                        above = self.cloud_density_grid[
                            self._cloud_index(int(sx), int(sy), int(sz))
                        ]
                        optical += above * 0.85
                    transmittance = math.exp(-optical)

                    height_factor = (z + 1) / nz
                    illuminated = ambient + direct_light * transmittance
                    illuminated *= 0.75 + 0.25 * height_factor
                    # Bases are dark: subtract light for voxels far below top.
                    illuminated *= 0.6 + 0.4 * height_factor
                    self.cloud_light_grid[idx] = illuminated * (0.4 + 0.6 * d)

        average_light = (
            sum(self.cloud_light_grid) / len(self.cloud_light_grid)
            if self.cloud_light_grid else 0.0
        )
        total_density = sum(self.cloud_density_grid)
        return {
            "density_grid": list(self.cloud_density_grid),
            "light_grid": list(self.cloud_light_grid),
            "shape": self.cloud_grid_shape,
            "coverage": coverage,
            "density": total_density / len(self.cloud_density_grid),
            "average_light": average_light,
            "wind_speed": speed,
            "time": self.cloud_time,
            "is_volumetric": True,
            "has_thickness": True,
            "shadowing": True,
        }

    # ------------------------------------------------------------------
    # Weather transitions
    # ------------------------------------------------------------------
    def transition_weather(
        self,
        target: Optional[str] = None,
        duration: float = 30.0,
        dt: float = 0.0,
        progress: Optional[float] = None,
    ) -> Dict[str, object]:
        """Blend cloud cover, light, fog and sky colour toward ``target``.

        The transition is a smooth (smoothstep) interpolation through
        continuous intermediate states rather than an instant switch.
        """
        if target is not None and target != self.weather_target_name:
            if target not in WEATHER_PRESETS:
                raise ValueError(f"Unknown weather preset: {target}")
            self.weather_target_name = target
            self.weather_target = dict(WEATHER_PRESETS[target])
            self._transition_from = dict(self.weather)
            self._transition_start = self.simulation_time
            self._transition_duration = max(duration, 1e-6)

        if dt:
            self.simulation_time += dt

        if progress is not None:
            t = _clamp(progress)
        elif self._transition_start is not None:
            elapsed = self.simulation_time - self._transition_start
            t = _clamp(elapsed / self._transition_duration)
        else:
            t = 1.0

        blend = t * t * (3.0 - 2.0 * t)
        for key in WEATHER_PARAMETERS:
            self.weather[key] = _mix(
                self._transition_from[key], self.weather_target[key], blend
            )

        sky_color = self.calculate_sky_color()
        if t >= 1.0:
            phase = self.weather_target_name
            finished = True
        else:
            phase = f"transition_{blend:.3f}"
            finished = False

        result = {
            "weather": phase,
            "target": self.weather_target_name,
            "progress": t,
            "finished": finished,
            "cloud_cover": self.weather["cloud_cover"],
            "cloud_density": self.weather["cloud_density"],
            "light_multiplier": self.weather["light_multiplier"],
            "sky_darkening": self.weather["sky_darkening"],
            "turbidity": self.weather["turbidity"],
            "fog_density": self.weather["fog_density"],
            "precipitation": self.weather["precipitation"],
            "sky_color": sky_color,
        }
        self.state["weather"] = dict(self.weather)
        return result
