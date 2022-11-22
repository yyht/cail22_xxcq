import os
import tensorflow as tf

def build_position_embeddings(embeddings, input_dim=512):
    position_ids = tf.range(0, 1024, dtype='int32')
    alpha = 0.4
    embeddings = embeddings - alpha * embeddings[:1]
    embeddings = embeddings / (1 - alpha)
    embeddings_x = tf.gather(embeddings, position_ids // input_dim)
    embeddings_y = tf.gather(embeddings, position_ids % input_dim)
    embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
    return embeddings

def re_ckpt(ckpt_dir, model_name, tf_checkpoint_path):
    tf_path = os.path.abspath(tf_checkpoint_path)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    
    sess = tf.Session()
    
    def create_tf_var(tensor, name, session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            continue
        if 'bert/embeddings/position_embeddings' in name:
            array = build_position_embeddings(array, input_dim=array.shape[0])
            array = sess.run(array)
        
        names.append(name)
        arrays.append(array)
        
    tf.reset_default_graph()
    with tf.Session() as session:
        for name, array in zip(names, arrays):
            if any(n in ["adam_v", "adam_m", "global_step"] for n in name.split('/')):
                continue
            tf_var = create_tf_var(tensor=array, name=name, session=session)
#             if isinstance(array, tf.Tensor):
#                 array = session.run(array)
            tf.keras.backend.set_value(tf_var, array)
            tf_weight = session.run(tf_var)

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name + ".ckpt"))

ckpt_dir = '/Users/xuhaotian/Downloads/bert_unilm_large'
model_name = 'bert_large_50g_mix_ilm_v2_model_840000'
tf_checkpoint_path = '/Users/xuhaotian/Downloads/roberta_base_mlm_unilm/bert_large_50g_mix_ilm_v2_model.ckpt-840000'
re_ckpt(ckpt_dir, model_name, tf_checkpoint_path)
    