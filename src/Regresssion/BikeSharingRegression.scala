package Regresssion.bikesharing

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
 * Author : lihuichuan
 * Time   : 2020/4/10
 **/
object BikeSharingRegression {
  def main(args: Array[String]): Unit = {

    // 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .master("local[6]")
      .appName("BikeSharingRegression")
      .getOrCreate()
    import spark.implicits._

    // 获取SparkContext实例对象
    val sc = spark.sparkContext
    // 设置日志级别
    sc.setLogLevel("WARN")

    // TODO 1. 读取CSV格式数据，首行为列名称
    val rawDF: DataFrame = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/Regresssion/bikesharing/hour.csv")


    // 样本数据及schema信息
    rawDF.printSchema()
    rawDF.show(10, truncate = false)



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

    // 获取特征标签向量
    val lpsRDD: RDD[LabeledPoint] = recordsRDD.map(row => {
      val categoryFeaturesArray = Array(
        // season            yr            mnth              hr
        row.getInt(0) - 1, row.getInt(1), row.getInt(2) - 1, row.getInt(3),
        // holiday       weekday       workingday     weathersit
        row.getInt(4), row.getInt(5), row.getInt(6), row.getInt(7) - 1
      ).map(_.toDouble)
      val otherFeaturesArray = Array(
        //  temp           atemp               hum               windspeed
        row.getDouble(8), row.getDouble(9), row.getDouble(10), row.getDouble(11)
      )
      // 特征想来了
      val features = Vectors.dense(categoryFeaturesArray ++ otherFeaturesArray)
      // 返回标签向量
      LabeledPoint(row.getInt(12).toDouble, features)
    })
    lpsRDD.take(10).foreach(println)

    /**
     * 无论是分类算法还是回归算法，算法中数据集都是RDD[LabeledPoint], LabeledPoint(label, features)
     */
    // TODO: 使用决策树回归算法训练模型， 对于决策树算法来说支持使用类别特征进行训练模型
    /*
  def trainRegressor(
    input: RDD[LabeledPoint],
    categoricalFeaturesInfo: Map[Int, Int],
    impurity: String,
    maxDepth: Int,
    maxBins: Int
  ): DecisionTreeModel
 */
    // 将数据集划分为 训练数据集和测试数据集 两部分
    val Array(testRDD, trainRDD) = lpsRDD.randomSplit(Array(0.2, 0.8), seed = 123L)
    println("================== 决策树回归算法 ==================")

    val dtrModel: DecisionTreeModel = DecisionTree.trainRegressor(
      trainRDD, // 训练数据集
      Map[Int, Int](
        0 -> 4, 1 -> 2, 2 -> 12, 3 -> 24, 4 -> 2, 5 -> 7, 6 -> 2, 7 -> 4
      ),
      "variance", // 回归模型中 不纯度度量方式仅有一种，就是方差
      10, // 默认值为5，表示树的最大深度
      64 // 最大分支数，必须大于等于 类别特征的 个数
    )
    // 使用模型预测

    val predictAndactualRDD: RDD[(Double, Double)] = testRDD.map(lp => {
      val predictLabel = dtrModel.predict(lp.features)
      (predictLabel, lp.label)
    })
    // 打印真实值和预测值比较
    predictAndactualRDD.take(10).foreach(println)

    val metrics = new RegressionMetrics(predictAndactualRDD)

    println(s"均方根误差RMSE = ${metrics.rootMeanSquaredError}")
    println(s"均方误差MSE = ${metrics.meanSquaredError}")
    println(s"平均绝对值误差 = ${metrics.meanAbsoluteError}")
    println(s"R2 = ${metrics.r2}")
    println(s"Variance = ${metrics.explainedVariance}")


    println("================== 随机森林回归算法 ==================")

    val rfrModel: RandomForestModel = RandomForest.trainRegressor(
      trainRDD,
      Map[Int, Int](
        0 -> 4, 1 -> 2, 2 -> 12, 3 -> 24, 4 -> 2, 5 -> 7, 6 -> 2, 7 -> 4
      ),
      10,
      "onethird",
      "variance",
      10,
      32
    )

    // 使用模型预测
    val rfPredictAndactualRDD: RDD[(Double, Double)] = testRDD.map(lp => {
      val predictLabel = rfrModel.predict(lp.features)
      (predictLabel, lp.label)
    })
    // 回归模型 评估指标
    // Instantiate metrics object   an RDD of (prediction, observation)
    val rfMmetrics = new RegressionMetrics(rfPredictAndactualRDD)
    println(s"使用随机森林算法得到模型的均方根误差RMSE = ${rfMmetrics.rootMeanSquaredError}")
    // 为了监控方便，让线程休眠一下
    Thread.sleep(1000000)
    // 关闭资源
    spark.stop()

  }
}
