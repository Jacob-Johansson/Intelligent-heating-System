
# Returns the heat from the heater, measured in watts (W = dQ_gain/dt).
def update(heatingpower, thermostatstate):
    return heatingpower * thermostatstate

def calculate_cost(heatingpower, heaterstate, costperjoule, dt):
    return heatingpower * heaterstate * costperjoule * dt