
def print_config(cfg):
    print("==== Loaded Config ====")
    print(f"Device: {cfg['device']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Max seq len: {cfg['max_seq_len']}")
    print("=======================\n")
