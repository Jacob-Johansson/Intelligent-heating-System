
# Returns the heat from the heater, measured in watts (W = dQ_gain/dt).
def Update(heatingPower, thermostatState):
    return heatingPower * thermostatState

def CalculateCost(heaterPower, heaterState, costPerJoule, dt):
    return heaterPower * heaterState * costPerJoule * dt