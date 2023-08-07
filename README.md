### 异常检测

* 异常检测部分代码位于data/wanguo文件夹下，文件树如下：

  ```
  .
  ├── anomoly_detect.py
  ├── data_process.py
  ├── group0_features.npy
  ├── group0_targets.npy
  ├── group1_features.npy
  ├── group1_targets.npy
  ├── group2_features.npy
  ├── group2_targets.npy
  ├── north
  │   ├── 202205北侧.xlsx
  │   ├── 202206北侧.xlsx
  │   ├── 202207北侧.xlsx
  │   ├── 202208北侧.xlsx
  │   ├── 202209北侧.xlsx
  │   ├── 202210北侧.xlsx
  │   ├── 202211北侧.xlsx
  │   ├── 202212北侧.xlsx
  │   ├── 202301北侧.xlsx
  │   ├── 202302北侧.xlsx
  │   ├── 202303北侧.xlsx
  │   └── 202304北侧.xlsx
  ├── south
  │   ├── 202207南侧.xlsx
  │   ├── 202208南侧.xlsx
  │   ├── 202209南侧.xlsx
  │   ├── 202210南侧.xlsx
  │   ├── 202211南侧.xlsx
  │   ├── 202212南侧.xlsx
  │   ├── 202301南侧.xlsx
  │   ├── 202302南侧.xlsx
  │   ├── 202303南侧.xlsx
  │   └── 202304南侧.xlsx
  ├── test
  │   └── soh.npy
  └── train
      └── soh.npy
  ```

* 原理：用时间序列分解算法，将电池单体的电压等数据中随机噪声与总体趋势分离。然后对大量正常的正常电池的趋势使用核密度估计，再对所有单体的当前时间点计算一个离群值，绘制出单体各自离群值曲线。同样对离群值曲线进行时间序列分解，获得离群值变化趋势，然后使用随机森林算法检测出异常曲线。

* 输入：首先使用data_process.py选择想要进行异常检测的特征（例如温度、电压等，仓库中默认的是电压），目前只支持同时对一个特征进行异常检测，若想要多个特征，可以使用不同的特征分别进行检测后，获得一个综合的评价结果），然后运行`python data_process.py`，会在目录下生成groupN_features.npy和groupN_targets.npy。由于数据中本身有三个电池簇，所以这里N的值为0、1、2。features.npy存储[m,1]的ndarray，第一维表示cycle输，第二维表示特征的值。

* 运行：

  ```
  python anomoly_detect.py -path [feature.npy的路径] -train [是否需要训练从而寻找合适的contamination值（随机森林算法使用，表示异常值占总样本的比例）] -epoch [训练轮数，配合train参数使用] -contamination [contamination预设值] -plot [是否绘图]
  ```

  准备好输入，可以先运行`python anomoly_detect.py -train True - epoch N`来获得一个较好的contamination值，然后在运行`python anomoly_detect.py -contamination x`来进行异常检测。

  全局变量：

  SEQ_LEN：计算离群值时，使用的曲线窗口大小。默认100cycle

  DEFAULT_OUTLIER_CURVE ：发生异常的曲线书目。默认1条曲线

* 输出：

  ```
  ERROR: [147]
  PREDICT: [147]
  precision: 1.0 recall: 1.0
  ```

  在代码中，会自动选取DEFAULT_OUTLIER_CURVE数量的曲线施加逐渐增大的噪声，使得值异常。输出结果中ERROR表示真正异常单体的编号，PREDICT表示算法检测出的单体编号。precision和recall分别表示算法的准确率和召回率。

### SOH预测

* 异常检测代码大多位于仓库根目录，文件树如下：

  ```
  .
  ├── __pycache__
  ├── data
  ├── data_aug.py
  ├── data_loader.py
  ├── dataset.py
  ├── figures
  ├── models.py
  ├── our_data_loader.py
  ├── reports
  ├── requirements.txt
  ├── rul_main.py
  ├── seq_sampler.py
  ├── soh_main.py
  ├── tool.py
  └── utils.py
  ```

* 原理：使用北侧的soh曲线，预测南侧的soh曲线。北侧的soh通过data_aug.py的插值采样进行数据增强后，获得若干条参考曲线。然后，按照每100个cycle为序列，将北侧和南侧的soh曲线分别切分为若干段，构成训练集和测试集。在训练时，通过当前序列的起始soh值寻找到参考曲线的对应参考点，并从参考点取同样长度的序列与原序列计算dtw，获得相似度。取topk个最相似的曲线，作为参考，将他们和原序列的相似度值经过MLP后计算一个权重值，并分别乘以它们在达到原序列后M周期的soh值所需的周期数，求和获取原序列后第M周期soh值所需的预测周期数，与M作差获得loss。测试时原理相同。

* 输入：测试集和训练集数据分别保存于根目录下data/wanguo/test和data/wanguo/train文件夹下的soh.npy中。其中存储的ndarray的形状为[m,1]，其中第一维m表示cycle数，第二维表示对应cycle的soh值。该数据可以通过data_process.py获得。

* 运行：

  ```
  usage: soh_main.py [-h] [-seq_len SEQ_LEN] [-N N] [-batch BATCH] [-valid_batch VALID_BATCH] [-num_worker NUM_WORKER] [-epoch EPOCH]
                     [-lr LR] [-top_k TOP_K] [-aug_path AUG_PATH] [-data_path DATA_PATH]
  
  options:
    -h, --help            show this help message and exit
    -seq_len SEQ_LEN      序列长度
    -N N                  预测当前序列末尾后N个cycle的soh值
    -batch BATCH          batch_size
    -valid_batch VALID_BATCH
                          batch_size
    -num_worker NUM_WORKER
                          number of worker
    -epoch EPOCH          num of epoch
    -lr LR                learning rate
    -top_k TOP_K          选择top_k个最相似的序列作为参考
    -aug_path AUG_PATH    数据增强使用的.npy文件的路径
    -data_path DATA_PATH  存有train和test文件夹的文件夹路径
  ```

  准备好数据后，可以直接运行`python soh_main.py`进行训练+测试。

* 输出：

  ```
  Epoch 7
  Start training
  train_average_loss 0.8806097408135732
  Start validating
  tensor(10.8354, grad_fn=<SumBackward0>) tensor(10.)
  tensor(10.0849, grad_fn=<SumBackward0>) tensor(10.)
  tensor(10.1431, grad_fn=<SumBackward0>) tensor(10.)
  tensor(9.9145, grad_fn=<SumBackward0>) tensor(10.)
  tensor(9.7786, grad_fn=<SumBackward0>) tensor(10.)
  test_average_loss 0.09580437342325847
  ```

这里的两个loss分别为该batch内训练集和测试集预测的周期数和实际值的平均差值。输出的tensor tensor则是测试集中实际的一些例子。第一个tensor为预测值，后一个为实际值。