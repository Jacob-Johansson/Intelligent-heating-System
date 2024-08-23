# Returns the heat loss, measured in watts (W = dQ_loss/dt).
def CalculateHeatLossRate(houseTemp, outTemp, thermalResistance):
    return (houseTemp - outTemp) / thermalResistance

# Returns rate of change of the house temperature, measured in celcius per second (C/s).
def CalculateHouseTempRate(airMass, airHeatCapacity, heatGainRate, heatLossRate):
    return (heatGainRate - heatLossRate) / (airMass * airHeatCapacity)

# Returns the new house temperature, measured in celcius (C).
def Update(houseTemp, outsideTemp, heatGainRate, thermalResistance, airMass, airHeatCapacity, dt):
    heatLossRate = CalculateHeatLossRate(houseTemp, outsideTemp, thermalResistance)
    houseTempRate = CalculateHouseTempRate(airMass, airHeatCapacity, heatGainRate, heatLossRate)
    return houseTemp + houseTempRate * dt