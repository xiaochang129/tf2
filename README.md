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
###    2.1. 数据处理
tf.data.Datesets：从磁盘中传输训练数据，数据集可迭代（但不是迭代器）。
@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    predict=model(x)
    ...
Keras.fit()API：
   model.compile(optimizer=optimizer, loss=loss_fn)
   model.fit(dataset)

###    2.2. Keras层
####   2.2.1 标准keras层
@tf.function
layers=[tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for hidden_size in [,,]]# layers[3].trainable_variables => returns [w3, b3]
perceptron = tf.keras.Sequential(layers)# perceptron.trainable_variables => returns [w0, b0, ...]
head = tf.keras.Sequential([...])
path = tf.keras.Sequential([perceptron, head])
####   2.2.2 自定义模型层
经典RNN
class DynamicRNN(tf.keras.Model):
    def __init__(self, rnn_cell):
      super(DynamicRNN, self).__init__(self)
      self.cell = rnn_cell
    def call(self, input_data):
      input_data = tf.transpose(input_data, [1, 0, 2])
      outputs = tf.TensorArray(tf.float32, input_data.shape[0])
      state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
      for i in tf.range(input_data.shape[0]):
        output, state = self.cell(input_data[i], state)
        outputs = outputs.write(i, output)
      return tf.transpose(outputs.stack(), [1, 0, 2]), state

# 训练模式
for x, y in data:
  with tf.GradientTape() as tape:
    prediction = path(x)
    loss = loss_fn_head(prediction, y)
  gradients = tape.gradient(loss, path.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path.trainable_variables))
# 保存参数
tf.saved_model.save(trunk, output_path)

        
###    2.5. 使用tf.metrics聚合数据和tf.summary来记录它
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
