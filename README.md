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
### 定义类
    tf.one_hot(labels, depth=3)
    tf.zeros([2, 3])，tf.ones(4)，tf.fill([2, 2], 9)  #[[9 9]，[9 9]]
    tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)，tf.truncated_normal()，tf.random.uniform() 均匀分布
    tf.Variable(tf.constant(5, dtype=tf.float32))
    tf.Variable(tf.random.normal([2,2], mean=0,stddev=1))
    tf.cast(张量名，dtype=数据类型)：强制tensor转换为该数据类型
    tf.convert_to_tensor(np.arange(0, 8), dtype=tf.int64)
    np.mgrid[：：,：：]生成等间隔数值点。生成二维数据，第一维 如果是2：5：1，则从2开始，步长1。
    tf.broadcast_to()： 成倍增加。
       a = [[1, 2, 3], [4, 5, 6]]
       b = [4, 6]
       tf.broadcast_to(a, b)  #[[1 2 3 1 2 3]，[4 5 6 4 5 6]，[1 2 3 1 2 3]，[4 5 6 4 5 6]]

### 计算类
    tf.multiply（）两个矩阵中对应元素各自相乘
        a, b = tf.constant([4.0,3.0]), tf.constant([8.0,10.])
        c=tf.multiply(a,b)
        print(c.numpy()))  # [32,30]
        print(c)           # tf.Tensor([32,30], shape=(2,2), dtype=float32)  
    tf.matmul（）将矩阵a乘以矩阵b，生成a * b。
    tf.reduce_min()：计算张量维度上元素的最小值  tf.reduce_max()，tf.reduce_mean()， tf.reduce_sum()
    tf.add() tf.subtract() tf.multiply() tf.divide() 加减乘除
    tf.pow() tf.square() tf.sqrt() 平方 次方 开方
    tf.nn.softmax(tf.constant([1.01, 2.01, -0.66]))
    squeeze() 删除维度  tf.expand_dims() 新增维度
        x = tf.constant([5.8, 4.0, 1.2, 0.2])  # （5）
        x1 = tf.expand_dims(x,axis=0)          #  (1,5)
        w1 = tf.constant([[-0.8, -0.34, -1.4],[0.6, 1.3, 0.25],[0.5, 1.45, 0.9],[0.65, 0.7, -1.2]])
        b1 = tf.constant([2.52, -3.1, 5.62])
        y = tf.matmul(x1, w1) + b1     #tf.Tensor([[ 1.0099998   2.008      -0.65999985]], shape=(1, 3), dtype=float32)
        y_dim = tf.squeeze(y)  # tf.Tensor([ 1.0099998   2.008      -0.65999985], shape=(3,), dtype=float32)
        y_pro = tf.nn.softmax(y_dim)  #tf.Tensor([0.2563381  0.69540703 0.04825491], shape=(3,), dtype=float32)
    assign_sub() 自减函数
        x = tf.Variable(4)
        x.assign_sub(1) # 4-1=3
    tf.math.top_k(output, maxk).indices： 最大maxk个元素的索引
             pred = tf.transpose(pred, perm=[1, 0])
             target_ = tf.broadcast_to(target, pred.shape)
             correct = tf.equal(pred, target_)

### 拼接类
    np.vstack() 拼接
        a = np.array([1, 2, 3]), b = np.array([4, 5, 6])
        c = np.vstack((a, b))    # [[1 2 3]， [4 5 6]]
    np.hstack() 水平拼接
        a = np.array([1, 2, 3]), b = np.array([4, 5, 6])
        c = np.hstack((a, b))    #[1 2 3 4 5 6]
    tf.transpose(input_x, perm=[0, 2, 1]): 转置，a[x,y,z]->a[x,z,y]
    tf.concat([tf.ones([1,2,3],tf.ones[4,2,3]],axis=0):   shape=[5,2,3]
        b = tf.constant([[1, 0, 2, 1]])		# 行组成的列表
        c = tf.constant([[0, 1, 1, 1]])     # 列组成的列表
        c_b = tf.transpose(tf.concat([b, c], axis=0))  #[[1 0]，[0 1]，[2 1]，[1 1]]
    tf.stack([a,b],axis=0):  shape相同的a,b 合并，并新增维度。[2,1,2,3]
        a,b=tf.unstack(c,axis=0)
    tf.argmax() 最大值索引
        test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
        tf.argmax(test, axis=0))
    tf.argsort()  排序索引，同样，还有tf.sort()
        id=tf.argsort(a)
        no=tf.gather(a,id)
    tf.greater(a, b) 判断大小,a的值大返回a，b的值大返回b。
    tf.where(tf.greater(a, b), a, b)  选择条件语句;类似于c语言 a>b?a:b 但是不完全一样，可以扩展到多维。
    tf.pad(a,[[x,y],[u,v],[s,t]]):在a矩阵的shape[0]前x行后y行，shape[1]的前u后v，shape[2]的前s后t插入0.
    tf.tile(a,[2,3]): 复制，shape[0]复制两次，shape[1]复制3次。
    tf.slice(input, begin, size, name = None)：是从输入数据input中提取出一块切片
         切片的开始位置是begin，切片的尺寸size表示输出tensor的数据维度，其中size[i]表示在第i维度上面的元素个数。
    tf.gather_nd(a, c_b)： 取出a中 c_b位置的变量。
    tf.split():  训练集、测试集切割。
### 模型上下文类  
    tf.data.Dataset.from_tensor_slices((输入特征， 标签))：将输入特征和标签进行匹配，构建数据集。
    tf.GradientTape()
    tf.losses.categorical_crossentropy() 交叉熵损失函数
        loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.7, 0.3]) # tf.Tensor(0.35667497, shape=(), dtype=float32)
    常用网络结构
        tf.matmul               tf.keras.layers.Dense
        tf.nn.conv2d                           .Conv2D
        tf.nn.relu                             .SimpleRNN
        tf.nn.max_pool2d                       .LSTM
        tf.nn.sigmoid                          .ReLU
        tf.nn.softmax                          .MaxPool2D

##  GPU 按需分配
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 设置使用哪一块GPU（默认是从0开始）

# 下面就是实现按需分配的代码！
         gpus = tf.config.experimental.list_physical_devices('GPU')
         if gpus:
           try:
             # Currently, memory growth needs to be the same across GPUs
             for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
           except RuntimeError as e:
             # Memory growth must be set before GPUs have been initialized
             print(e)
 ##  一个样例
          # monitor监听器, 连续5个验证准确率不增加，这个事情触发。
         # early_stopping：当验证集损失值，连续增加小于0时，持续10个epoch，则终止训练。
         early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                        min_delta=0.00001,
                                                        patience=10, verbose=1)
         # reduce_lr：当评价指标不在提升时，减少学习率，每次减少10%，当验证损失值，持续3次未减少时，则终止训练。
         reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                                                       patience=10, min_lr=0.000001, verbose=1)
         # 网络的装配。
         newnet.compile(optimizer=optimizers.Adam(lr=1e-4), loss=losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
         # 完成标准的train，val, test; 标准的逻辑必须通过db_val挑选模型的参数，就需要提供一个earlystopping技术，
         newnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=500,
                    callbacks=[early_stopping, reduce_lr])   # 1个epoch验证1次。触发了这个事情，提前停止了。
         newnet.evaluate(db_test)

