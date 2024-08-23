import environment, simulation.importer as importer, qstar, qPolicies
import numpy as np
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

def SegmentHourTempsInto6MinuteTemps(temps):
    # Segment the data temps, in hours, into 6-minute inverval.
    tempsMinutes = np.zeros(len(temps) * 10)
    for i in range(0, len(temps) - 1):
        for j in range(0, 10):
            # Lerp the temperature
            tempsMinutes[i * 10 + j] = temps[i] + (temps[i + 1] - temps[i]) * j / 10
    return tempsMinutes

env = environment.HouseEnvironment()
env.OutdoorTemps = SegmentHourTempsInto6MinuteTemps(importer.ImportHuskvarna())
env.TargetTemps = np.ones(len(env.OutdoorTemps))*21
env.TempDifferenceHysteresis = 2

time_step = env.reset()

cumulative_reward = time_step.reward

numEpisodes = 10000
hysteresis = 2
minTempDifference = -hysteresis
maxTempDifference = hysteresis
tempDifferenceStep = 0.25
numTempDifferenceStates = int((maxTempDifference - minTempDifference) / tempDifferenceStep) + 1

decayInitialValue = 1.0
decayRate = 0.0001
decayMinValue = 0.01

qTable = np.random.rand(numTempDifferenceStates, numTempDifferenceStates, env.Heater.NumStates) * 0.01
learningRate = 0.1
discountFactor = 0.9
print('Q-Table shape: ', qTable.shape)

rewards = np.zeros(numEpisodes)
steps = np.zeros(numEpisodes)

minDiff = 10000
maxDiff = -10000

for _ in range(numEpisodes):
    
    # Calculate the exploration probability per episode.
    explorationProbability = qstar.CalculateExponentialDecay(decayInitialValue, decayRate, decayMinValue, _)
    
    # Calculate the initial state.
    tempDifferenceA = time_step.observation[0]
    tempDifferenceB = time_step.observation[1]
    currentStateIndexA = int((tempDifferenceA - minTempDifference) / tempDifferenceStep)
    currentStateIndexB = int((tempDifferenceB - minTempDifference) / tempDifferenceStep)
    
    episodeReward = 0
    episodeSteps = 0
    while not time_step.is_last() and (currentStateIndexA >= 0 and currentStateIndexA < numTempDifferenceStates and currentStateIndexB >= 0 and currentStateIndexB < numTempDifferenceStates):
        
        # Get the current action.
        currentAction = qPolicies.EpsiloneGreedyPolicy(qTable, [currentStateIndexA, currentStateIndexB], env.action_spec().shape[0], explorationProbability)
        
        # Step the environment.
        time_step = env.step(currentAction)
        episodeReward += time_step.reward
        
        # Calculate the next state.
        nextTempDifferenceA = time_step.observation[0]
        nextTempDifferenceB = time_step.observation[1]
        nextStateIndexA = int((nextTempDifferenceA - minTempDifference) / tempDifferenceStep)
        nextStateIndexB = int((nextTempDifferenceB - minTempDifference) / tempDifferenceStep)
        nextStateIndexB = min(numTempDifferenceStates - 1, max(0, nextStateIndexB))
        
        minDiff = min(minDiff, nextTempDifferenceB)
        maxDiff = max(maxDiff, nextTempDifferenceB)
        
        # Update the Q-table.
        if nextStateIndexA >= 0 and nextStateIndexA < numTempDifferenceStates:
            qMax = np.max(qTable[nextStateIndexA, nextStateIndexB, :])
            qTable[currentStateIndexA, currentStateIndexB, currentAction] = (1 - learningRate) * qTable[currentStateIndexA, currentStateIndexB, currentAction] + learningRate * (time_step.reward + discountFactor * qMax)
            currentStateIndexA = nextStateIndexA
            currentStateIndexB = nextStateIndexB
        else:
            qTable[currentStateIndexA, currentStateIndexB, currentAction] += -1
            episodeReward += -1
            break
        
        episodeSteps += 1
    
    # Reset the environment for the next episode.
    time_step = env.reset()

    rewards[_] = episodeReward / episodeSteps if episodeSteps > 0 else 1
    steps[_] = episodeSteps
    
    print('Episode: ', _, ' Reward = ', rewards[_], ' Steps = ', steps[_])

print('Min diff: ', minDiff, ' Max diff: ', maxDiff)
temps = importer.ImportHuskvarna()
# Segment the data temps, in hours, into 6-minute inverval.
tempsMinutes = SegmentHourTempsInto6MinuteTemps(temps)
        
targetTemps = np.ones(len(tempsMinutes)) * 21
[roomTempResult, targetTempResult, outTempResult, costResult, heatGainResult, states] = qstar.Simulate(tempsMinutes, targetTemps, env.GetAirMass(), env.GetAirHeatCapacity(), env.GetThermalResistance(), env.Heater.GetTotalPower(), env.GetHeatingCost(), 20, qTable, minTempDifference, tempDifferenceStep, numTempDifferenceStates, env.Heater.NumStates)

plt.figure()
plt.plot(rewards)

plt.figure()
plt.plot(steps)

plt.figure()
plt.plot(roomTempResult)
plt.plot(targetTempResult)
plt.plot(outTempResult)
plt.legend(['Indoor Temperature', 'Target Temperature', 'Outside Temperature'])

# Show the qtale.
plt.figure()
tempDifferences = np.linspace(minTempDifference, maxTempDifference, numTempDifferenceStates)
plt.plot(tempDifferences, qTable[:, 0])
plt.plot(tempDifferences, qTable[:, 1])
plt.plot(tempDifferences, qTable[:, 2])
plt.plot(tempDifferences, qTable[:, 3])
plt.plot(tempDifferences, qTable[:, 4])
plt.legend(['Heater Off', 'Heater Low', 'Heater Medium', 'Heater High', 'Heater Max'])

plt.show()