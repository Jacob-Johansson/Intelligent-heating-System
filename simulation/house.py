# Returns the heat loss, measured in watts (W = dQ_loss/dt).
def calculate_heat_loss_rate(indoortemperature, outdoortemperature, thermalresistance):
    return (indoortemperature - outdoortemperature) / thermalresistance

# Returns rate of change of the indoor temperature, measured in celcius per second (C/s).
def calculate_indoor_temperature_rate(airmass, airheatcapacity, heatgainrate, heatlossrate):
    return (heatgainrate - heatlossrate) / (airmass * airheatcapacity)

# Returns the new indoor temperature, measured in celcius (C).
def update(indoortemperature, outdoortemperature, heatgainrate, thermalresistance, airmass, airheatcapacity, dt):
    heatLossRate = calculate_heat_loss_rate(indoortemperature, outdoortemperature, thermalresistance)
    houseTempRate = calculate_indoor_temperature_rate(airmass, airheatcapacity, heatgainrate, heatLossRate)
    return indoortemperature + houseTempRate * dt