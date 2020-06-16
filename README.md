##  1.被替换的tf1和替换的tf2
- tf.app、tf.flags和tf.logging，tf.contrib     - tf.summary, tf.keras.metrics和tf.keras.optimizers。
- session.run()                                - outputs = f(input)  （在tf.function中，带有副作用的代码按写入的顺序执行）
- tf.Variable                                  - keras机制

##  2. 流程
###    2.1 数据处理
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
###    2.2 模型层
####   2.2.1 标准keras层
    @tf.function
    layers=[tf.keras.layers.Dense(size, activation=tf.nn.sigmoid) for size in [..]]# layers[3].trainable_variables => returns [w3, b3]
    perceptron = tf.keras.Sequential(layers)# perceptron.trainable_variables => returns [w0, b0, ...]
    head = tf.keras.Sequential([..])
    path = tf.keras.Sequential([perceptron, head])
####   2.2.2 自定义模型层
    build的作用是参数化输入输出维度。
##### 在init方法（keras.Model中自带build）
    class myDense (layers.Layer):
        def __init__(self, in_dim, out_dim):   
            super(myDense, self).__init__()
            self.kernel = self.add_variable('w', [in_dim, out_dim])
            self.bias = self.add_variable('b', [out_dim])
        def call(self, inputs, training = None):
            out = inputs @ self.kernel + self.bias
            return out
    class Model (keras.Model):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = myDense(32*32*3 ,256)   
            self.fc2 = myDense(256, 128)
        def call(self, inputs, training = None):
            inputs = tf.reshape(inputs, [-1, 32 * 32 * 3])
            x = self.fc1(inputs)
            out = tf.nn.relu(x)
            x = self.fc2(out)
            return x
##### build方法
       class SegPosEmbedding(tf.keras.layers.Layer):
        def __init__(self,**kwargs):
            super(SegPosEmbedding, self).__init__(name=name, **kwargs)
            self.use_token_type = use_token_type
        def build(self, input_shape):
            self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
            self.token_type_table = self.add_weight(name=,shape=[self.token_type_vocab_size, input_shape[2]],..)
            self.full_position_embeddings = self.add_weight(name=,shape=[self.max_position_embeddings, input_shape[2]],)
            self.drop_out = tf.keras.layers.Dropout(self.hidden_dropout_prob)
            self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="LayerNorm")
            self.built = True    #系统默认值为False，调用build之前先判断self.built是否完成。
        def call(self, input_tensor, token_type_ids=None, is_training=True):
            token_type_embeddings = tf.gather(self.token_type_table, token_type_ids)
            input_tensor += token_type_embeddings
            output = self.layer_norm(input_tensor)
            output = self.drop_out(output, training=is_training)
            return output
    
####   2.3  训练/预测
####   2.3.0  实例化
    model = Model(param)
    model.build(input_shape=(3, param.batch_size, param.maxlen))
    model.summary()
    optimizer = optim.AdamWarmup(learning_rate=1e-5) #设置优化器
    albert_init_weights_from_checkpoint(model,model_path, param.num_hidden_layers,pooler=False) #可以没有
####   2.3.1  全量训练
    model.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.1), 
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), metrics = ['sparse_categorical_accuracy'])
    model.fit(x_train, y_train,batch_size = 32,epochs = 500,validation_split = 0.2,validation_freq = 20)
    model.predict
####   2.3.2  batch训练
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        y_batch_pred = Model(x_batch_train)
        loss = loss_cal(y_batch_pred, y_batch_train)
      gradients = tape.gradient(loss, Model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, Model.trainable_variables))
###   2.4 保存参数
    tf.saved_model.save(Model, output_path)
###   2.5 记录过程
        train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
        with train_summary_writer.as_default():
          for images, labels in dataset:
            loss = loss_cal(model(images), labels)
            avg_loss.update_state(loss)                                               #累计值
            if tf.equal(optimizer.iterations % log_freq, 0):
              tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations) # .result()返回累计结果
              avg_loss.reset_states()                                                 #.reset_states()清除累计值。
###   2.6 TensorBoard指向摘要日志：
    tensorboard --logdir /tmp/summaries
