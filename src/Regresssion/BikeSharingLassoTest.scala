package Regresssion

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LassoWithSGD, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

/**
  * 使用Spark MLlib中回归算法预测共享单车每小时出租的次数
  * 使用线性回归算法训练模型
  */
object BikeSharingLassoTest {

  def main(args: Array[String]): Unit = {

    // 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("BikeSharingLassoTest")
      .getOrCreate()
    import spark.implicits._

    // 获取SparkContext实例对象
    val sc = spark.sparkContext
    // 设置日志级别
    sc.setLogLevel("WARN")

    // TODO 1. 读取CSV格式数据，首行为列名称
    val rawDF: DataFrame = spark.read
      .option("header", "true")
      //  .option("inferSchema", "true")
      .csv("file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/Regresssion/bikesharing/hour.csv")

    // 样本数据及schema信息
    rawDF.printSchema()
    rawDF.show(10, truncate = false)

    /**
      * root
      * |-- instant: integer (nullable = true)   序号，从1开始，可以不问
      * |-- dteday: timestamp (nullable = true) 年月日时分秒日期格式
      * |-- season: integer (nullable = true)  季节
      * 1 = spring, 2 = summer, 3 = fall, 4 = winter
      * |-- yr: integer (nullable = true)  年份
      * 2011 = 0, 2012 = 1
      * |-- mnth: integer (nullable = true) 月份
      * 1 - 12
      * |-- hr: integer (nullable = true)  小时
      * 0 -23
      * |-- holiday: integer (nullable = true) 是否是节假日
      * 要么是0 要么是1
      * |-- weekday: integer (nullable = true)  一周第几天
      * *
      * |-- workingday: integer (nullable = true)  是否是工作日
      * 要么是0 要么是1
      * |-- weathersit: integer (nullable = true)  天气状况
      * 1， 2， 3， 4
      * |-- temp: double (nullable = true)  气温
      * |-- atemp: double (nullable = true)  体感温度
      * |-- hum: double (nullable = true)  湿度
      * |-- windspeed: double (nullable = true) 方向
      * 上述四个特征值 经过 正则化处理以后的数据
      * |-- casual: integer (nullable = true)
      * 没有注册的用户租用自行车的数量
      * |-- registered: integer (nullable = true)
      * 注册的用户租用自行的数量
      * |-- cnt: integer (nullable = true)
      * 总的租用自行车的数量
      */


    /**
      * 将类别的特征转换为数值，数组表示形式
      *
      * @param categoryMap
      * 类别对应的Map集合
      * @param categoryValue
      * 某个类别值
      * @return
      */
    def transformCategory(categoryMap: Map[String, Long], categoryValue: String): Array[Double] = {
      // a. 构建数组，长度为类别特征长度（有多少个类别值）
      val categoryArray: Array[Double] = new Array[Double](categoryMap.size)
      // 依据类别的值获取对应的Map集合中Value的值（index）
      val categoryIndex: Long = categoryMap(categoryValue)
      // 赋予对应数组下标的值为1.0
      categoryArray(categoryIndex.toInt) = 1.0
      // 返回数组
      categoryArray
    }


    // TODO: 2. 选取特征值
    /**
      * 特征工程：
      *   a. 选取特征
      *   b. 特征处理（类别特征处理、特征归一化或标准化或正则化、数值转换等等）
      */
    val recordsRDD: RDD[Row] = rawDF
      .select(
        // 8个类别特征值 , 全部都是 integer 类型
        $"season", $"yr", $"mnth", $"hr", $"holiday", $"weekday", $"workingday", $"weathersit",
        // 4个数值特征值， 全部都是 double 类型
        $"temp", $"atemp", $"hum", $"windspeed",
        // 标签值：预测的值，integer 类型
        $"cnt"
      ).rdd

    // TODO: 获取 8 个类别特征对应Map集合映射（类别值 -> 索引）
    val categoryMapMap = (0 to 7).map(index => {
      // 分别对索引下标的值获取，构建Map集合映射
      val categoryMap = recordsRDD
        .map(row => row.getString(index)).distinct()
        .zipWithIndex().collectAsMap()
      // 以二元组形式返回，转换为Map集合
      (index, categoryMap)
    }).toMap
    // 通过广播变量将Map集合广播到各个Executor中
    val categoryMapMapBroadcast = sc.broadcast(categoryMapMap)

    // 获取特征标签向量
    val lpsRDD: RDD[LabeledPoint] = recordsRDD.map(row => {
      // 获取广播变量的值
      val categoryMapMapValue = categoryMapMapBroadcast.value

      val arrayBuffer = new ArrayBuffer[Double]()
      for (idx <- 0 to 7) {
        arrayBuffer ++= transformCategory(categoryMapMapValue(idx), row.getString(idx))
      }
      arrayBuffer += row.getString(8).toDouble
      arrayBuffer += row.getString(9).toDouble
      arrayBuffer += row.getString(10).toDouble
      arrayBuffer += row.getString(11).toDouble

      // 特征
      val features = Vectors.dense(arrayBuffer.toArray)
      // 返回标签向量
      LabeledPoint(row.getString(12).toDouble, features)
    })
    lpsRDD.take(10).foreach(println)

    // 将数据集划分为 训练数据集和测试数据集 两部分
    val Array(testRDD, trainRDD) = lpsRDD.randomSplit(Array(0.2, 0.8), seed = 123L)
    // 由于迭代多次训练模型，缓存训练数据集
    trainRDD.cache()
    testRDD.cache()

    /**
      * 无论是分类算法还是回归算法，算法中数据集都是RDD[LabeledPoint], LabeledPoint(label, features)
      */
    // 假设类别特征数据没有进行转换，直接使用
    //   def train(input: RDD[LabeledPoint], numIterations: Int): LinearRegressionModel
    // TODO: RMSE = 144.61504167460197
    // val lrModel = LassoWithSGD.train(trainRDD, 10)

    // TODO: 设置迭代次数为100   RMSE = 114.86942034091221
    // val lrModel = LassoWithSGD.train(trainRDD, 100)

    // TODO: 设置正则化参数位0.001    RMSE = 114.8274018166971
    // val lrModel = LassoWithSGD.train(trainRDD, 100, 1.0, 0.001, 1.0)

    // TODO: 设置正则化参数位0.1   RMSE = 115.29935507303477
    val lrModel = LassoWithSGD.train(trainRDD, 100, 1.0, 0.1, 1.0)

    // 使用模型预测
    val predictAndActualRDD: RDD[(Double, Double)] = testRDD.map(lp => {
      val predictLabel = lrModel.predict(lp.features)
      (predictLabel, lp.label)
    })
    // 打印真实值和预测值比较
    predictAndActualRDD.take(10).foreach(println)

    // 回归模型 评估指标
    // Instantiate metrics object   an RDD of (prediction, observation)
    val metrics = new RegressionMetrics(predictAndActualRDD)
    println(s"均方根误差RMSE = ${metrics.rootMeanSquaredError}")
    println(s"平均绝对值误差MAE = ${metrics.meanAbsoluteError}")

    // 为了监控方便，让线程休眠一下
    Thread.sleep(1000000)
    // 关闭资源
    spark.stop()
  }
}
