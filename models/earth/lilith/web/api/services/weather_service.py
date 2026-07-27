import os
import time
import httpx
from typing import Dict, Any, Optional
from loguru import logger

class WeatherService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 900  # 15 minutes TTL

    async def get_current_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Fetch current weather from OpenWeatherMap with caching.
        Returns a dict with 'temp', 'temp_min', 'temp_max', 'description', 'icon', 'dt'.
        """
        cache_key = f"{lat:.2f},{lon:.2f}"
        
        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry['timestamp'] < self._cache_ttl:
                return entry['data']
        
        # Fetch from API
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                # Extract relevant fields
                weather_data = {
                    "temp": data["main"]["temp"],
                    "temp_min": data["main"]["temp_min"],
                    "temp_max": data["main"]["temp_max"],
                    "description": data["weather"][0]["description"],
                    "icon": data["weather"][0]["icon"],
                    "dt": data["dt"]
                }
                
                # Update cache
                self._cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": weather_data
                }
                return weather_data
                
        except Exception as e:
            logger.error(f"Failed to fetch weather for {lat}, {lon}: {e}")
            return None

    async def get_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Fetch 5-day forecast from OpenWeatherMap.
        """
        cache_key = f"forecast_{lat:.2f},{lon:.2f}"
        
        # Check cache (longer TTL for forecast: 1 hour)
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry['timestamp'] < 3600:
                return entry['data']
        
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            async with httpx.AsyncClient() as client:
                # OWM 5 day / 3 hour forecast endpoint
                response = await client.get("https://api.openweathermap.org/data/2.5/forecast", params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                # Update cache
                self._cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": data
                }
                return data
                
        except Exception as e:
            logger.error(f"Failed to fetch forecast for {lat}, {lon}: {e}")
            return None
