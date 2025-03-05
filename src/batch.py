from netpyne.batch import Batch
from netpyne import specs
import numpy as np

# ----------------------------------------------------------------------------------------------
# Adaptive Stochastic Descent (ASD)
# ----------------------------------------------------------------------------------------------
def optunaRates():
    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    # long-range inputs
    params[('weightLong', 'TPO')] = [0.25, 0.75]
    params[('weightLong', 'TVL')] = [0.25, 0.75]
    params[('weightLong', 'S1')] = [0.25, 0.75]
    params[('weightLong', 'S2')] = [0.25, 0.75]
    params[('weightLong', 'cM1')] = [0.25, 0.75]
    params[('weightLong', 'M2')] = [0.25, 0.75]
    params[('weightLong', 'OC')] = [0.25, 0.75]

    # EEgain
    params['EEGain'] = [0.5, 1.5]

    # IEgain
    ## L2/3+4
    params[('IEweights', 0)] = [0.5, 1.5]
    ## L5
    params[('IEweights', 1)] = [0.5, 1.5]  # [0.8, 1.0]
    ## L6
    params[('IEweights', 2)] = [0.5, 1.5]  # [0.8, 1.0]

    # IIGain
    ## L2/3+4
    params[('IIweights', 0)] = [0.5, 1.5]
    ## L5
    params[('IIweights', 1)] = [0.5, 1.5]  # [0.8, 1.0]
    ## L6
    params[('IIweights', 2)] = [0.5, 1.5]  # [0.8, 1.0]

    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg['duration'] = 1500
    initCfg['printPopAvgRates'] = [500, 1500]
    initCfg['dt'] = 0.025

    initCfg['scaleDensity'] = 1.0

    # cell params
    initCfg['ihGbar'] = 0.75  # ih (for quiet/sponti condition)
    initCfg['ihModel'] = 'migliore'  # ih model
    initCfg['ihGbarBasal'] = 1.0  # multiplicative factor for ih gbar in PT cells
    initCfg['ihlkc'] = 0.2  # ih leak param (used in Migliore)
    initCfg['ihLkcBasal'] = 1.0  # multiplicative factor for ih lk in PT cells
    initCfg['ihLkcBelowSoma'] = 0.01  # multiplicative factor for ih lk in PT cells
    initCfg['ihlke'] = -86  # ih leak param (used in Migliore)
    initCfg['ihSlope'] = 28  # ih leak param (used in Migliore)

    initCfg['somaNa'] = 5.0  # somatic Na conduct
    initCfg['dendNa'] = 0.3  # dendritic Na conduct (reduced to avoid dend spikes)
    initCfg['axonNa'] = 7  # axon Na conduct (increased to compensate)
    initCfg['axonRa'] = 0.005
    initCfg['gpas'] = 0.5
    initCfg['epas'] = 0.9

    # long-range input params
    initCfg['numCellsLong'] = 1000
    initCfg[('pulse', 'pop')] = 'None'
    initCfg[('pulse', 'start')] = 1000.0
    initCfg[('pulse', 'end')] = 1100.0
    initCfg[('pulse', 'noise')] = 0.8

    # conn params
    initCfg['IEdisynapticBias'] = None

    initCfg['weightNormThreshold'] = 4.0
    initCfg['IEGain'] = 1.0
    initCfg['IIGain'] = 1.0
    initCfg['IPTGain'] = 1.0

    # plotting and saving params
    initCfg[('analysis', 'plotRaster', 'timeRange')] = initCfg['printPopAvgRates']
    initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False

    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}

    ## Exc pops
    Epops = ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']

    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    for pop in Epops:
        pops[pop] = Etune

    ## Inh pops
    Ipops = ['NGF1', 'PV2', 'SOM2', 'VIP2', 'NGF2',
             'PV4', 'SOM4', 'VIP4', 'NGF4',
             'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
             'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
             'PV6', 'SOM6', 'VIP6', 'NGF6']

    Itune = {'target': 10, 'width': 15, 'min': 0.25}
    for pop in Ipops:
        pops[pop] = Itune

    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000

    def fitnessFunc(simData, **kwargs):
        import numpy as np
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        popFitness = [min(np.exp(abs(v['target'] - simData['popRates'][k]) / v['width']), maxFitness)
                      if simData['popRates'][k] > v['min'] else maxFitness for k, v in pops.items()]
        fitness = np.mean(popFitness)

        popInfo = '; '.join(
            ['%s rate=%.1f fit=%1.f' % (p, simData['popRates'][p], popFitness[i]) for i, p in enumerate(pops)])
        print('  ' + popInfo)
        return fitness

    # from IPython import embed; embed()

    b = Batch(cfgFile='src/cfg.py', netParamsFile='src/netParams.py', params=params, groupedParams=groupedParams, initCfg=initCfg)
    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc,  # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters': 1e6,  # Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime': None,  # Maximum time allowed, in seconds
        'maxiter_wait': 16,
        'time_sleep': 60,
        'popsize': 1  # unused - run with mpi
    }

    return b

# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_bulletin'):
    if type=='mpi_bulletin' or type=='mpi':
        b.runCfg = {'type': 'mpi_bulletin',
            'script': 'src/init.py',
            'skip': True}

    elif type == 'hpc_sge_evol':
        b.runCfg = {'type': 'hpc_sge',
                    'jobName': 'M1_CR',
                    'cores': 19,
                    'log': os.getcwd() + '/' + b.batchLabel + '.log',
                    'mpiCommand': 'mpiexec',
                    'vmem': '90G',
                    'walltime': "15:00:00",
                    'script': 'src/init.py',
                    'queueName': 'cpu.q',
                    'skip': False}

# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    b = optunaRates()
    b.batchLabel = 'v104_batch1'
    b.saveFolder = 'batchData/'+b.batchLabel
    setRunCfg(b, 'mpi_bulletin')
    b.run() # run batch