目录结构：1.TF1和2的区别
         2.数据处理
         3.模型层设计
         4.实例化
         5.训练
         6.常用函数
##  1.TF1和2的区别
- tf.app、tf.flags和tf.logging，tf.contrib     #### tf.summary, tf.keras.metrics和tf.keras.optimizers。
- session.run()                                #### outputs = f(input)  （在tf.function中，带有副作用的代码按写入的顺序执行）
- tf.Variable                                  #### keras机制

##  2 数据处理
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
##  3 模型层设计
### 3.1 标准keras层
    @tf.function
    layers=[tf.keras.layers.Dense(size, activation=tf.nn.sigmoid) for size in [..]]# layers[3].trainable_variables => returns [w3, b3]
    perceptron = tf.keras.Sequential(layers)# perceptron.trainable_variables => returns [w0, b0, ...]
    head = tf.keras.Sequential([..])
    path = tf.keras.Sequential([perceptron, head])
### 3.2 自定义模型层
    build的作用是参数化输入输出维度。
#### 3.2.1 在init方法（keras.Model中自带build）
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
#### 3.2.2 build方法
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
    
##   4  实例化
    model = Model(param)
    model.build(input_shape=(3, param.batch_size, param.maxlen))
    model.summary()
    optimizer = optim.AdamWarmup(learning_rate=1e-5) #设置优化器
    model.load_weights(checkpoint_save_path) #可以没有，或另写。
##   5.训练
###  5.1  全量训练
    model.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.1), 
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), metrics = ['sparse_categorical_accuracy'])
    model.fit(x_train, y_train,batch_size = 32,epochs = 500,validation_split = 0.2,validation_freq = 20)
    model.predict
###  5.2  batch训练
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        y_batch_pred = Model(x_batch_train)
        loss = loss_cal(y_batch_pred, y_batch_train)
      gradients = tape.gradient(loss, Model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, Model.trainable_variables))
###  5.3 保存参数
     tf.saved_model.save(Model, output_path)
###  5.4 记录过程
     train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
     with train_summary_writer.as_default():
       for images, labels in dataset:
         loss = loss_cal(model(images), labels)
         avg_loss.update_state(loss)                                               #累计值
         if tf.equal(optimizer.iterations % log_freq, 0):
           tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations) # .result()返回累计结果
           avg_loss.reset_states()                                                 #.reset_states()清除累计值。
###  5.5 TensorBoard指向摘要日志：
    tensorboard --logdir /tmp/summaries

##   6.常用函数
    1.tf.multiply（）两个矩阵中对应元素各自相乘
        a, b = tf.constant([4.0,3.0]), tf.constant([8.0,10.])
        c=tf.multiply(a,b)
        print(c.numpy()))  # [32,30]
        print(c)           # tf.Tensor([32,30], shape=(2,2), dtype=float32)  
    2.tf.matmul（）将矩阵a乘以矩阵b，生成a * b。
    3.tf.cast(张量名，dtype=数据类型)：强制tensor转换为该数据类型
    4.tf.reduce_min()：计算张量维度上元素的最小值  tf.reduce_max()，tf.reduce_mean()， tf.reduce_sum()
    5.tf.Variable(tf.random.normal([2,2], mean=0,stddev=1))
    6.tf.data.Dataset.from_tensor_slices((输入特征， 标签))：将输入特征和标签进行匹配，构建数据集。
    7.tf.GradientTape()
    8.tf.one_hot(labels, depth=3)
    9.tf.zeros([2, 3])，tf.ones(4)，tf.fill([2, 2], 9)  #[[9 9]，[9 9]]
    10.tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)，tf.truncated_normal()，tf.random.uniform() 均匀分布
    11.tf.Variable(tf.constant(5, dtype=tf.float32))
    12.tf.convert_to_tensor(np.arange(0, 8), dtype=tf.int64)
    13. tf.add() tf.subtract() tf.multiply() tf.divide() 加减乘除
    14. tf.pow() tf.square() tf.sqrt() 平方 次方 开方
    15. tf.nn.softmax(tf.constant([1.01, 2.01, -0.66]))
    16.tf.squeeze() 删除
        x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])  # 5.8,4.0,1.2,0.2（0）
        w1 = tf.constant([[-0.8, -0.34, -1.4],
                          [0.6, 1.3, 0.25],
                          [0.5, 1.45, 0.9],
                          [0.65, 0.7, -1.2]])
        b1 = tf.constant([2.52, -3.1, 5.62])
        y = tf.matmul(x1, w1) + b1     #tf.Tensor([[ 1.0099998   2.008      -0.65999985]], shape=(1, 3), dtype=float32)
        y_dim = tf.squeeze(y)  # tf.Tensor([ 1.0099998   2.008      -0.65999985], shape=(3,), dtype=float32)
        y_pro = tf.nn.softmax(y_dim)  #tf.Tensor([0.2563381  0.69540703 0.04825491], shape=(3,), dtype=float32)
    17.assign_sub() 自减函数
        x = tf.Variable(4)
        x.assign_sub(1) # 4-1=3
    18.tf.argmax() 最大值索引
        test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
        tf.argmax(test, axis=0))
    19.tf.greater(a, b) 判断大小,a的值大返回a，b的值大返回b。
    20.tf.where(tf.greater(a, b), a, b)  选择条件语句;类似于c语言 a>b?a:b 但是不完全一样，可以扩展到多维。
    21. np.vstack() 拼接
        a = np.array([1, 2, 3]), b = np.array([4, 5, 6])
        c = np.vstack((a, b))    # [[1 2 3]， [4 5 6]]
    22. np.hstack() 水平拼接
        a = np.array([1, 2, 3]), b = np.array([4, 5, 6])
        c = np.hstack((a, b))    #[1 2 3 4 5 6]
    23. np.mgrid[：：,：：]生成等间隔数值点。生成二维数据，第一维 如果是2：5：1，则从2开始，步长1 。
    24. tf.losses.categorical_crossentropy() 交叉熵损失函数
        loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.7, 0.3]) # tf.Tensor(0.35667497, shape=(), dtype=float32)
    25.常用网络结构
        tf.matmul               tf.keras.layers.Dense
        tf.nn.conv2d                           .Conv2D
        tf.nn.relu                             .SimpleRNN
        tf.nn.max_pool2d                       .LSTM
        tf.nn.sigmoid                          .ReLU
        tf.nn.softmax                          .MaxPool2D
