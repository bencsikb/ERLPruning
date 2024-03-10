import os

def document_model_details(model, path):
    """
    Documents the different layers in a given model: layer type, size.
    """

    if os.path.exists(path):
        print(f"This file already exists!")
        #return 0

    with open(path, "w") as f:

        network_size = len(model.module_list)
        f.write(f"Network size: {network_size}\n")
        for i in range(network_size):

            sequential_size = len(model.module_list[i])
            module_def = model.module_defs[i]
            #f.write(f"Layer {i}\n{module_def}\n")
            f.write(f"Layer {i}\n")

            if module_def["type"] == "convolutional":
                for j in range(sequential_size):
                    layer = model.module_list[i][j]
                    f.write(f"{layer}\n")






