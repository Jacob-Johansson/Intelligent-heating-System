
def Update(targetTemp, currentTemp, hysteresis, currentState):
    if currentTemp > targetTemp + hysteresis:
        return 0
    elif currentTemp < targetTemp - hysteresis:
        return 1
    else:
        return currentState