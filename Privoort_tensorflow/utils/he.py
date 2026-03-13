import os, numpy as np, tenseal as ts
from collections import OrderedDict
import tensorflow as tf

def get_ckks_context(dir_path=".ckks_context", name="context",
                     poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=(60,40,40,60),
                     global_scale=2**40):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return ts.context_from(f.read())
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=poly_modulus_degree,
                     coeff_mod_bit_sizes=list(coeff_mod_bit_sizes))
    ctx.global_scale = global_scale
    with open(path, "wb") as f:
        f.write(ctx.serialize(save_secret_key=True))
    return ctx

# def flatten_weights(state_dict: dict[str, torch.Tensor]):
#     flat = np.concatenate([w.flatten().cpu().numpy() for w in state_dict.values()])
#     shapes = {k: v.shape for k, v in state_dict.items()}
#     sizes = {k: v.numel() for k, v in state_dict.items()}
#     return flat, shapes, sizes

def flatten_weights(model: tf.keras.Model):
    weights = model.trainable_variables
    flat_list = [tf.reshape(w, [-1]).numpy() for w in weights]
    flat = np.concatenate(flat_list).astype(np.float32) if flat_list else np.array([], dtype=np.float32)
    #shapes = {w.shape for w in weights}
    shapes = [tuple(w.shape.as_list()) for w in weights]
    #sizes = {w.shape.num_elements() for w in weights}
    sizes = [int(np.prod(s)) for s in shapes]
    return flat, shapes, sizes

# def rebuild_weights(vec, shapes, sizes):
#     out, offset = OrderedDict(), 0
#     for k, n in sizes.items():
#         slice_ = vec[offset:offset+n].reshape(shapes[k])
#         out[k] = torch.tensor(slice_)
#         offset += n
#     return out

def rebuild_weights(vec, shapes, sizes):
    out = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        slice_ = vec[offset:offset+size].reshape(shape)
        out.append(slice_)
        offset += size
    return out

def encrypt_vector(vec, ctx):
    return ts.ckks_vector(ctx, vec).serialize()

def decrypt_vector(blob, ctx):
    v = ts.lazy_ckks_vector_from(blob)
    v.link_context(ctx)
    return np.array(v.decrypt())

def load_encrypted_vector(blob, ctx):
    vec = ts.lazy_ckks_vector_from(blob)
    vec.link_context(ctx)
    return vec  # CKKSVector，可直接做加权求和

# def serialize_context(ctx, dir_path, name):
#     os.makedirs(dir_path, exist_ok=True)
#     with open(os.path.join(dir_path, name), "wb") as f:
#         f.write(ctx.serialize(save_secret_key=True))

