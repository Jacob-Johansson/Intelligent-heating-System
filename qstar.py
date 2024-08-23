import numpy as np
import random
import simulation.importer as importer, simulation.simulation as simulation
import matplotlib.pyplot as plt

def CalculateReward(targetTemp, newTemp):
    return 1 / (1 + np.abs(newTemp - targetTemp))

def CalculateRewardV2(targetTemp, newTemp, previousTemp, dt):
    ra = -np.abs(newTemp - targetTemp)
    rb = min(0.5, -np.abs((newTemp - previousTemp) / 1))
    return 1*ra

def NormalizeReward(reward, minReward, maxReward):
    if maxReward == minReward:
        return -1
    
    return -1 + (reward - minReward) / (maxReward - minReward)

def IsValidState(currentState, numIndoorTemps, numOutdoorTemps):
    return currentState[0] >= 0 and currentState[0] < numIndoorTemps and currentState[1] >= 0 and currentState[1] < numOutdoorTemps

def IsValidState(currentState, numTempStates):
    return currentState[0] >= 0 and currentState[0] < numTempStates
    
def CalculateQValueForAction(nextStateReward, qMaxNextState:float, currentStateQValueForAction, learningRate, discountRate:float):
    return learningRate * (nextStateReward + discountRate * qMaxNextState - currentStateQValueForAction)

# Returns the value of the exponential decay function, used to calculate the exploration rate at each time step.
def CalculateExponentialDecay(initialValue, decayRate, minDecayValue, time):
    return max(initialValue * np.exp(-decayRate * time), minDecayValue)

# Returns a discretized state space in numIndoorTemps * numOutdoorTemps * numHeaterStates.
def DiscretizeStateSpace(numIndoorTemps, numOutdoorTemps, numOutdoorTempDerivatives, numHeaterStates):
    stateSpace = np.zeros((numIndoorTemps, numOutdoorTemps, numOutdoorTempDerivatives, numHeaterStates))
    return stateSpace

def DiscretizeStateSpace(numTempStates, numOutdoorDerivativeStates, numHeaterStates):
    return np.zeros((numTempStates, numOutdoorDerivativeStates, numHeaterStates))

# Returns the index of the state space for the given indoor and outdoor temperatures.
def CalculateStateSpaceIndex(tempDifference, minTempDifference, tempDifferenceStep, outdoorDerivative, minOutdoorDerivative, outdoorDerivativeStep, numOutdoorDerivativeStates):
    outdoorIndex = min(numOutdoorDerivativeStates - 1, max(0, int((outdoorDerivative - minOutdoorDerivative) / outdoorDerivativeStep)))
    return [int((tempDifference - minTempDifference) / tempDifferenceStep), outdoorIndex]

# Returns the action based on the Epsilon-Greedy policy.
def SelectActionIndex(qTable, currentState, numHeaterStates, explorationRate):
    
    # Select a random action with probability explorationRate
    if np.random.rand() < explorationRate:
        return np.random.randint(0, numHeaterStates)
    # Select the action with the highest Q-value
    else:
        return np.argmax(qTable[currentState[0], :])

# Returns the heater state based on the action index.
def GetHeaterStateFromActionIndex(actionIndex, numHeaterStates):
    return actionIndex / (numHeaterStates - 1)

def CalculateTemperatureDifference(targetTemp, currentTemp):
    return targetTemp - currentTemp
    
def GetDayOrNight(time):
    if time < 6 or time > 22:
        return 0
    else:
        return 1

def Simulate(outsideTemps, targetTemps, airMass, airHeatCapacity, thermalResistance, heaterPower, costPerJoule, initialHouseTemp, qTable, minTempDifference, tempDifferenceStep, numTempStates, numHeaterStates):
    
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
    prevHouseTemp = houseTemp
    
    for i in range(0, len(outsideTemps)):
        outsideTemp = outsideTemps[i]
        
        targetTemp = targetTemps[i]
        
        # Samples are taken every 6 minute.
        for j in range(0, 1):
            index = i
            targetTempResult[index] = targetTemp
            roomTempResult[index] = houseTemp
            outTempResult[index] = outsideTemp
            costResult[index] = totalCost
            heatGainResult[index] = totalHeatGain
            states[index] = thermostatState
            
            tempDifferenceA = CalculateTemperatureDifference(targetTemp, houseTemp)
            tempDifferenceB = CalculateTemperatureDifference(houseTemp, prevHouseTemp)
            
            currentStateIndexA = int((tempDifferenceA - minTempDifference) / tempDifferenceStep)
            currentStateIndexB = int((tempDifferenceB - minTempDifference) / tempDifferenceStep)
            currentStateIndexA = min(numTempStates - 1, max(0, currentStateIndexA))
            currentStateIndexB = min(numTempStates - 1, max(0, currentStateIndexB))
        
            # Get the action with the highest Q-value
            action = np.argmax(qTable[currentStateIndexA, currentStateIndexB, :])
            heaterState = GetHeaterStateFromActionIndex(action, numHeaterStates)

            StepResult = simulation.Step(outsideTemp, houseTemp, airMass, airHeatCapacity, thermalResistance, heaterPower, heaterState, costPerJoule, 360)
            
            prevHouseTemp = houseTemp
            
            houseTemp = StepResult[0]
            totalHeatGain += StepResult[1]
            totalCost += StepResult[2]
            
            
 
    return [roomTempResult, targetTempResult, outTempResult, costResult, heatGainResult, states]