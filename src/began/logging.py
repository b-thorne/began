import os
import datetime
from pathlib import Path
import tensorflow as tf

def setup_vae_run_logging(LAT_DIM, BATCH_SIZE, EPOCHS):
    """ Function to setup the logging for a given run. This requires the
    environment variable TF_LOGDIR to exist. If it does not exist an error
    will be thrown. 
    
    Parameters
    ----------
    
    hyperparameters: dict
        Dictionary containing the hyperparameters for this run. These are
        latent dimension, batch size, and number of training epochs.
        
    Notes
    -----
    Will create logging directory if it does not exist.
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = Path(os.environ['TF_LOGDIR']) 
    network_identifier = "LATDIM-{:03d}_BATCHSZ-{:03d}_EPOCHS-{:03d}".format(LAT_DIM, BATCH_SIZE, EPOCHS)
    network_logdir = logging_dir / network_identifier
    network_logdir.mkdir(exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(network_logdir / current_time))
    return summary_writer