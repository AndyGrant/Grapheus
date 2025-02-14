import numpy as np

n_king_buckets  = 32
n_relations     = 10
n_squares       = 64

n_real_features    = n_king_buckets * n_relations * n_squares
n_virtual_features = n_relations * n_squares
n_features         = n_real_features + n_virtual_features

n_l0 = 768
n_l1 = 8
n_l2 = 32
n_l3 = 1

quant_ft = 64
quant_l1 = 32
quant_l2 = 1
quant_l3 = 1

def largest(arr, N):
    return np.sort(arr)[-100:]

def smallest(arr, N):
    return np.sort(arr)[:100]

def read_array(file, dtype, dims):
    rows, cols = dims
    size = np.dtype(dtype).itemsize * rows * cols
    data = np.frombuffer(file.read(size), dtype=dtype)
    return data if rows == 1 else data.reshape((rows, cols))

def write_array(file, array, dtype):
    array.astype(dtype).tofile(file)

if __name__ == '__main__':

    with open('run1_WDL/weights/500.state', 'rb') as fin:

        dims = {
            'ft_weights' : (n_features, n_l0), 'ft_biases'  : (1, n_l0),
            'l1_weights' : (n_l0 * 2  , n_l1), 'l1_biases'  : (1, n_l1),
            'l2_weights' : (n_l1      , n_l2), 'l2_biases'  : (1, n_l2),
            'l3_weights' : (n_l2      , n_l3), 'l3_biases'  : (1, n_l3),
        }

        ft_weights = read_array(fin, np.float32, dims['ft_weights'])
        ft_biases  = read_array(fin, np.float32, dims['ft_biases' ])
        l1_weights = read_array(fin, np.float32, dims['l1_weights'])
        l1_biases  = read_array(fin, np.float32, dims['l1_biases' ])
        l2_weights = read_array(fin, np.float32, dims['l2_weights'])
        l2_biases  = read_array(fin, np.float32, dims['l2_biases' ])
        l3_weights = read_array(fin, np.float32, dims['l3_weights'])
        l3_biases  = read_array(fin, np.float32, dims['l3_biases' ])

    virtual_weights   = ft_weights[-n_virtual_features:]
    real_weights      = ft_weights[:-n_virtual_features]
    collapsed_weights = real_weights.copy()

    for i in range(0, n_real_features, n_relations * n_squares):
        collapsed_weights[i:i+n_relations * n_squares] += virtual_weights

    ft_biases         = [round(f * quant_ft) for f in ft_biases        .flatten() ]
    collapsed_weights = [round(f * quant_ft) for f in collapsed_weights.flatten() ]
    l1_biases         = [round(f * quant_l1) for f in l1_biases        .flatten() ]
    l1_weights        = [round(f * quant_l1) for f in l1_weights       .flatten() ]
    l2_biases         = [       f * quant_l2 for f in l2_biases        .flatten() ]
    l2_weights        = [       f * quant_l2 for f in l2_weights       .flatten() ]
    l3_biases         = [       f * quant_l3 for f in l3_biases        .flatten() ]
    l3_weights        = [       f * quant_l3 for f in l3_weights       .flatten() ]

    with open('exported.nn', 'wb') as fout:
        write_array(fout, np.array(ft_biases        ), np.int16  )
        write_array(fout, np.array(collapsed_weights), np.int16  )
        write_array(fout, np.array(l1_biases        ), np.int32  )
        write_array(fout, np.array(l1_weights       ), np.int8   )
        write_array(fout, np.array(l2_biases        ), np.float32)
        write_array(fout, np.array(l2_weights       ), np.float32)
        write_array(fout, np.array(l3_biases        ), np.float32)
        write_array(fout, np.array(l3_weights       ), np.float32)

