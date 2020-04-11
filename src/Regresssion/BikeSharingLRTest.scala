package Regresssion

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable.ArrayBuffer

/**
 * Author : lihuichuan
 * Time   : 2020/4/11
 **/
object BikeSharingLRTest {

  def main(args: Array[String]): Unit = {
    // 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("BikeSharingLRTest")
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

    def transformCategory(categoryMap: collection.Map[String, Long], categoryValue: String): Array[Double] = {
      val categoryArray: Array[Double] = new Array[Double](categoryMap.size)

      val categoryIndex: Long = categoryMap(categoryValue)
      categoryArray(categoryIndex.toInt) = 1.0
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

    val categoryMapMap: Map[Int, collection.Map[String, Long]] = (0 to 7).map(index => {
      val categoryMap = recordsRDD.map(row => {
        row.getString(index)
      }).distinct()
        .zipWithIndex().collectAsMap()
      (index, categoryMap)
    }).toMap

    // 通过广播变量将Map集合广播到各个Executor中
    val categoryMapMapBroadcast = sc.broadcast(categoryMapMap)

    // 获取特征标签向量
    val lpsRDD = recordsRDD.map(row => {
      // 获取广播变量的值
      val categoryMapMapValue = categoryMapMapBroadcast.value
      val arrayBuffer = new ArrayBuffer[Double]()
      for (idx <- 0 to 7) {
        arrayBuffer ++= transformCategory(categoryMapMapBroadcast.value(idx), row.getString(idx))
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

    // TODO：设置步长为0.1    RMSE = 159.78709974573084
    val lrModel: LinearRegressionModel = LinearRegressionWithSGD.train(
      trainRDD, 100, 0.1, 1.0)

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
