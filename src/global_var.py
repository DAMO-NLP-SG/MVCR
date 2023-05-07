def init():
    global last_epoch_training
    global kl_loss
    global recon_loss
    global pretrain_ae
    pretrain_ae = False
    last_epoch_training = False
    kl_loss = [None, None, None, None, None, None, None, None, None, None, None, None]
    recon_loss = [None, None, None, None, None, None, None, None, None, None, None, None]
