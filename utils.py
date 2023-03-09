def calculateReceptiveField(res_layers, res_blocks, kernal_size):
    dialtions = [2**i for i in range(res_layers)] * res_blocks
    return (kernal_size - 1) * sum(dialtions) + 1 