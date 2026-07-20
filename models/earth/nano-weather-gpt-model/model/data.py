"""Generate synthetic weather report text for training the nanoGPT."""

import random


def generate_weather_corpus(n_samples: int = 5000, seed: int = 42) -> str:
    random.seed(seed)

    conditions = [
        "sunny", "partly cloudy", "mostly cloudy", "overcast", "rainy",
        "heavy rain", "light drizzle", "thunderstorms", "foggy", "misty",
        "snowy", "light snow", "clear", "windy", "breezy",
    ]
    cities = [
        "Zurich", "Bern", "Geneva", "Basel", "Lausanne",
        "Lugano", "Lucerne", "St. Gallen", "Winterthur", "Zug",
    ]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]

    templates = [
        "Today in {city}, the weather is {condition} with a temperature of {temp} degrees Celsius. "
        "Humidity stands at {hum} percent with winds at {wind} kilometers per hour from the {dir}.",

        "Weather forecast for {city} on {day}: expect {condition} skies. "
        "Temperature will reach {temp} degrees Celsius with {hum} percent humidity.",

        "Good morning! {day} brings {condition} weather to {city}. "
        "The temperature is {temp} degrees and humidity is at {hum} percent.",

        "The forecast for {month} shows {condition} conditions with temperatures around {temp} degrees. "
        "Wind speeds are expected at {wind} km/h from the {dir}.",

        "In {city} today, {condition} skies dominate. "
        "Temperatures hover around {temp} degrees with moderate humidity of {hum} percent.",

        "Weather update: {condition} conditions are expected in {city}. "
        "Temperature drops to {temp} degrees with {hum} percent humidity and {wind} km/h winds.",

        "Tomorrow's forecast: {condition} with temperatures reaching {temp} degrees. "
        "Humidity at {hum} percent, winds at {wind} km/h from the {dir}.",

        "Current conditions in {city}: {condition}. Temperature: {temp} degrees Celsius. "
        "Humidity: {hum} percent. Wind: {wind} km/h {dir}.",

        "This {day} in {city}: {condition} weather expected. "
        "Temperatures of {temp} degrees and humidity levels of {hum} percent.",

        "Meteorologists predict {condition} weather for the coming days in {city}. "
        "Temperatures ranging from {temp_low} to {temp_high} degrees Celsius.",

        "The {month} weather report shows {condition} skies over {city}. "
        "Expect {temp} degrees Celsius with winds blowing from the {dir} at {wind} km/h.",

        "Weekend weather: {city} will experience {condition} conditions. "
        "High of {temp_high} degrees, low of {temp_low} degrees, humidity around {hum} percent.",

        "Aviation weather for {city}: {condition} at ground level. "
        "Temperature {temp} degrees, dewpoint {dew} degrees, wind {dir} at {wind} km/h.",
    ]

    lines = []
    for _ in range(n_samples):
        template = random.choice(templates)
        temp = random.randint(-10, 38)
        temp_high = temp + random.randint(2, 8)
        temp_low = temp - random.randint(2, 8)
        dew = temp - random.randint(3, 12)
        line = template.format(
            city=random.choice(cities),
            condition=random.choice(conditions),
            temp=temp,
            temp_high=temp_high,
            temp_low=temp_low,
            dew=dew,
            hum=random.randint(20, 98),
            wind=random.randint(0, 90),
            dir=random.choice(directions),
            day=random.choice(days),
            month=random.choice(months),
        )
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    corpus = generate_weather_corpus()
    with open("weather_corpus.txt", "w") as f:
        f.write(corpus)
    print(f"Corpus saved: {len(corpus):,} characters")
