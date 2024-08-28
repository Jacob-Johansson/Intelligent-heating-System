import house, heater, thermostat
import matplotlib.pyplot as plt
import numpy as np

#def Step(outsideTemp, houseTemp, airMass, airHeatCapacity, thermalResistance, heaterPower, heaterState, costPerJoule, integrateSeconds):
#    
#    heatGain = 0
#    costGain = 0
#    
#    dt = 1/1 # dt is same time unit as the parameters (thermalReistance, etc).
#    
#    # Integrate the system for the specified time.
#    for t in range(0, integrateSeconds):
#        heatGainRate = heater.Update(heaterPower, heaterState)
#        houseTemp = house.Update(houseTemp, outsideTemp, heatGainRate, thermalResistance, airMass, airHeatCapacity, dt)
#
#        heatGain += heatGainRate * dt
#    
#    costGain = heater.CalculateCost(heaterPower, heaterState, costPerJoule, integrateSeconds) #heatGain * costPerJoule
#    return [houseTemp, heatGain, costGain]
#
#def StepV2(outsideTemp, houseTemp, airMass, airHeatCapacity, thermalResistance, heaterPower, heaterState, costPerJoule, integrateSeconds):
#    heatGain = heaterPower * heaterState * integrateSeconds
#    costGain = heaterPower * heaterState * costPerJoule * integrateSeconds
#    
#    # dt is same time unit as the parameters (thermalReistance, etc).
#    Mdot = airMass * airHeatCapacity
#    k1 = thermalResistance * Mdot
#    k2 = (thermalResistance * heaterPower * heaterState) + outsideTemp
#    
#    for t in range(0, integrateSeconds):
#        houseTemp = ((k1-1)/k1) * houseTemp + ((k2/k1))
#    
#    #houseTemp = ((k1-1)/k1) * houseTemp * (integrateSeconds - 1) + ((k2/k1)*integrateSeconds)
#    return [houseTemp, heatGain, costGain]
#

# Returns the new indoor temperature, heat gain and cost gain.
def step(outdoortemperature, indoortemperature, airmass, airheatcapacity, thermalresistance, heatingpower, heaterstate, costperjoule, integrateseconds):
    # Calculate heat gain and cost
    heatgain = heatingpower * heaterstate * integrateseconds
    costgain = heatgain * costperjoule
    
    # Calculate the thermal time constant (tau)
    tau = thermalresistance * airmass * airheatcapacity
    
    # Exponential decay factor
    exp_factor = np.exp(-integrateseconds / tau)
    
    # Calculate new house temperature
    indoortemperature = (outdoortemperature + (indoortemperature - outdoortemperature) * exp_factor 
                 + heatingpower * heaterstate * thermalresistance * (1 - exp_factor))
    
    return [indoortemperature, heatgain, costgain]


def simulate(outdoortemperatures, targettemperatures, airmass, airheatcapacity, thermalresistance, heatingpower, costperjoule, initialindoortemperature, hysteresis, dt):
    
    numsamples = len(outdoortemperatures)
    indoortemperatures = np.zeros(numsamples)
    costs = np.zeros(numsamples)
    heatgains = np.zeros(numsamples)
    heaterstates = np.zeros(numsamples)
    
    indoortemperature = initialindoortemperature
    totalcost = 0
    thermostatstate = 0
    
    totalheatgain = 0
    
    for i in range(0, len(outdoortemperatures)):
        outdoortemperature = outdoortemperatures[i]
        
        targettemperature = targettemperatures[i]

        indoortemperatures[i] = indoortemperature
        costs[i] = totalcost
        heatgains[i] = totalheatgain
        heaterstates[i] = thermostatstate
    
        thermostatstate = thermostat.update(targettemperature, indoortemperature, hysteresis, thermostatstate)
        StepResult = step(outdoortemperature, indoortemperature, airmass, airheatcapacity, thermalresistance, heatingpower, thermostatstate, costperjoule, dt)
        
        indoortemperature = StepResult[0]
        totalheatgain += StepResult[1]
        totalcost += StepResult[2]
 
    return [indoortemperatures, targettemperatures, outdoortemperatures, costs, heatgains, heaterstates]