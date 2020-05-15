def adjust_learning_rate(optimizer, epoch, lr):
    """sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * ((1 - 0.015) ** epoch)
    print('learning rate : {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
