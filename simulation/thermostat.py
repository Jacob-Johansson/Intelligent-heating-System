
def update(targettemperature, indoortemperature, hysteresis, thermostatstate):
    if indoortemperature > targettemperature + hysteresis:
        return 0
    elif indoortemperature < targettemperature - hysteresis:
        return 1
    else:
        return thermostatstate