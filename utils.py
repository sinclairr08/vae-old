def to_gpu(var, is_gpu):
    return var.cuda() if is_gpu else var