from netpyne.batchtools.search import search

params = {'weightLong.TPO': [0.25, 0.75],
          'weightLong.S1': [0.25, 0.75],
          'weightLong.S2': [0.25, 0.75],
          'weightLong.cM1': [0.25, 0.75],
          'weightLong.M2': [0.25, 0.75],
          'weightLong.OC': [0.25, 0.75],
          'EEGain': [0.5, 1.5],
          'IEweights.0': [0.5, 1.5],
          'IEweights.1': [0.5, 1.5],
          'IEweights.2': [0.5, 1.5],
          'IIweights.0': [0.5, 1.5],
          'IIweights.1': [0.5, 1.5],
          'IIweights.2': [0.5, 1.5],
          }

# use batch_shell_config if running directly on the machine
shell_config = {'command': 'python -u init.py'}

# use batch_sge_config if running on a
sge_config = {
    'queue': 'cpu.q',
    'cores': 5,
    'vmem': '4G',
    'realtime': '00:30:00',
    'command': 'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi init.py'}


run_config = shell_config

search(job_type = 'sh', # or sh
       comm_type = 'socket',
       label = 'optuna',
       params = params,
       output_path = '../batchData/optuna_batch',
       checkpoint_path = '../batchData/ray',
       run_config = run_config,
       num_samples = 10,
       metric = 'loss',
       mode = 'min',
       algorithm = 'optuna',
       max_concurrent = 1)