def config_args(args, conf):
    if conf == 1:
        args.model = 'vae'
        args.dataset = 'mnist'
        args.ninput = 784
        args.nlatent = 20
        args.nhidden = 400
        args.lr_ae = 1e-3

    elif conf == 2:
        args.model = 'aae'
        args.dataset = 'mnist'
        args.ninput = 784
        args.nlatent = 120
        args.nhidden = 1000
        args.nDhidden = 500
        args.lr_ae = 1e-04
        args.lr_disc = 5e-05

    elif conf == 3:
        args.model = 'arae'
        args.dataset = 'mnist'
        args.ninput = 784
        args.nlatent = 120
        args.nhidden = 1000
        args.nDhidden = 500
        args.nGhidden = 500

    elif conf == 4:
        args.model = 'vqvae'
        args.dataset = 'mnist'
        args.nlatent = 64
        args.nembdim = 64
        args.nemb = 512
        args.ninput = 784
        args.nhidden = 400
        args.lr_ae = 1e-3
        args.commit_cost = 1


    elif conf == 5:
        args.model = 'lstmvae'
        args.dataset = 'snli'
        args.nlatent = 300
        args.nhidden = 300


    elif conf == 6:
        args.model = 'lstmaae'
        args.dataset = 'snli'
        args.nlatent = 300
        args.nDhidden = 300
        args.niters_gan_d = 5

    elif conf == 7:
        args.model = 'lstmarae'
        args.dataset = 'snli'
        args.nlatent = 300
        args.nnoise = 100
        args.nDhidden = 300
        args.nGhidden = 300
        args.niters_gan_d = 5

    elif conf == 8:
        args.model = 'lstmvqvae'
        args.dataset = 'snli'
        args.nlatent = 300
        args.nembdim = 300
        args.nemb = 512

    else:
        raise NameError

    return args