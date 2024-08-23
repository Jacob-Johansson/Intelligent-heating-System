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

# Returns the new house temperature, heat gain and cost gain.
def Step(outsideTemp, houseTemp, airMass, airHeatCapacity, thermalResistance, heaterPower, heaterState, costPerJoule, integrateSeconds):
    # Calculate heat gain and cost
    heatGain = heaterPower * heaterState * integrateSeconds
    costGain = heatGain * costPerJoule
    
    # Calculate the thermal time constant (tau)
    tau = thermalResistance * airMass * airHeatCapacity
    
    # Exponential decay factor
    exp_factor = np.exp(-integrateSeconds / tau)
    
    # Calculate new house temperature
    houseTemp = (outsideTemp + (houseTemp - outsideTemp) * exp_factor 
                 + heaterPower * heaterState * thermalResistance * (1 - exp_factor))
    
    return [houseTemp, heatGain, costGain]


def Simulate(outsideTemps, targetTemps, airMass, airHeatCapacity, thermalResistance, heaterPower, targetTemp, costPerJoule, initialHouseTemp, hysteresis):
    
    numSamples = len(outsideTemps)
    roomTempResult = np.zeros(numSamples)
    outTempResult = np.zeros(numSamples)
    targetTempResult = np.ones(numSamples)
    costResult = np.zeros(numSamples)
    heatGainResult = np.zeros(numSamples)
    states = np.zeros(numSamples)
    
    houseTemp = initialHouseTemp
    totalCost = 0
    thermostatState = 0
    
    totalHeatGain = 0
    
    for i in range(0, len(outsideTemps)):
        outsideTemp = outsideTemps[i]
        
        targetTemp = targetTemps[i]
        targetTempResult[i] = targetTemp

        # Samples are taken every 6 minute.
        for j in range(0, 1):
            index = i
            roomTempResult[index] = houseTemp
            outTempResult[index] = outsideTemp
            costResult[index] = totalCost
            heatGainResult[index] = totalHeatGain
            states[index] = thermostatState
        
            thermostatState = thermostat.Update(targetTemp, houseTemp, hysteresis, thermostatState)
            StepResult = Step(outsideTemp, houseTemp, airMass, airHeatCapacity, thermalResistance, heaterPower, thermostatState, costPerJoule, 360)
            
            houseTemp = StepResult[0]
            totalHeatGain += StepResult[1]
            totalCost += StepResult[2]
 
    return [roomTempResult, targetTempResult, outTempResult, costResult, heatGainResult, states]