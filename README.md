##  1.tf的变动
链接：https://www.jianshu.com/p/599c79c3a537
###    1.1被替换的tf1
- tf.app、tf.flags和tf.logging，tf.contrib
- session.run()
- tf.Variable

###    1.2替换的tf2
- tf.summary, tf.keras.metrics和tf.keras.optimizers。
- outputs = f(input)  （在tf.function中，带有副作用的代码按写入的顺序执行）
- keras机制

##  2. tf2的写法
###    2.1 数据处理
tf.data.Datesets：从磁盘中传输训练数据，数据集可迭代（但不是迭代器）。

###    2.2 模型层
####   2.2.1 标准keras层
@tf.function
layers=[tf.keras.layers.Dense(size, activation=tf.nn.sigmoid) for size in [..]]# layers[3].trainable_variables => returns [w3, b3]
perceptron = tf.keras.Sequential(layers)# perceptron.trainable_variables => returns [w0, b0, ...]
head = tf.keras.Sequential([..])
path = tf.keras.Sequential([perceptron, head])
####   2.2.2 自定义模型层
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
####   2.3  训练/预测
####   2.3.1  全量训练
model.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.1), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), metrics = ['sparse_categorical_accuracy'])
model.fit(x_train, y_train,batch_size = 32,epochs = 500,validation_split = 0.2,validation_freq = 20)
model.predict
####   2.3.2  batch训练
for x, y in data:
  with tf.GradientTape() as tape:
    y_pred = Model(x)
    loss = loss_cal(y_pred, y)
  gradients = tape.gradient(loss, Model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, Model.trainable_variables))
####  2.4 保存参数
tf.saved_model.save(Model, output_path)
###   2.5. tf.metrics聚合数据和tf.summary记录
#第六步，model.summary()
model.summary() #打印神经网络结构，统计参数数目
        tf.summary.(scalar|histogram|...)：记录摘要。
            summary_writer = tf.summary.create_file_writer('/tmp/summaries')
            with summary_writer.as_default():
              tf.summary.scalar('loss', 0.1, step=42)
        tf.metrics，Metrics：聚合数据，当你调用.result()时，它们会累计值并返回累计结果。使用.reset_states()清除累计值。
            def train(model, optimizer, dataset, log_freq=10):
              avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
              for images, labels in dataset:
                loss = train_step(model, optimizer, images, labels)
                avg_loss.update_state(loss)
                if tf.equal(optimizer.iterations % log_freq, 0):
                  tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
                  avg_loss.reset_states()
            def test(model, test_x, test_y, step_num):
              loss = loss_fn(model(test_x), test_y)
              tf.summary.scalar('loss', loss, step=step_num)
            train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
            test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')
            with train_summary_writer.as_default():
              train(model, optimizer, dataset)
              test(model, test_x, test_y, optimizer.iterations)
        TensorBoard指向摘要日志：
            tensorboard --logdir /tmp/summaries
