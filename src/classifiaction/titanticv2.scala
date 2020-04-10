package classifiaction

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy, Strategy}
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.loss.SquaredError
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author : lihuichuan
 * Time   : 2020/4/9
 **/
object titanticv2 {

  def main(args: Array[String]): Unit = {
    // TODO: 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .appName("TitanicClassificationMLTest")
      .master("local[*]")
      .getOrCreate()
    // 导入隐式转换
    import spark.implicits._

    // 获取SparkContext实例对象
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    /**
     * TODO: a. 读取泰坦尼克号数据集
     */
    val titanicDF: DataFrame = spark.read
      .option("header", "true").option("inferSchema", "true")
      .csv("file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/classifiaction/titanic/train.csv")
    // 样本数据
    titanicDF.show(10, truncate = false)
    titanicDF.printSchema()

    /**
     * TODO: b. 特征工程，提取特征值，组合到标签向量中LabeledPoint
     */

    val avgAge: Double = titanicDF.select($"Age").agg("Age" -> "avg").first().getDouble(0)

    // TODO: 获取性别Sex集合映射Map
    val sexCategoryMap = titanicDF
      .select($"Sex").rdd
      .map(row => row.getString(0)).distinct()
      .zipWithIndex().collectAsMap()

    // 使用广播变量将数据集合Map广播
    val sexCategoryMapBroadcast = sc.broadcast(sexCategoryMap)

    sexCategoryMap.foreach(println)

    val titanicRDD: RDD[LabeledPoint] = titanicDF.select(
      $"Survived", $"Pclass", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare"
    ).rdd.map(row => {
      val label = row.getInt(0).toDouble

      // TODO: 针对Sex特征进行处理：把Sex变量的取值male替换为1，female替换为0
      val sexFeature = new Array[Double](sexCategoryMapBroadcast.value.size)
      sexFeature(sexCategoryMapBroadcast.value(row.getString(2)).toInt) = 1.0

      // TODO: 针对Age特征进行转换：有117个乘客年龄值有缺失，用平均年龄30岁替换
      val ageFeature = if (row.get(3) == null) avgAge else row.getDouble(3)

      // 获取特征值
      val features = Vectors.dense(
        Array(row.getInt(1).toDouble, ageFeature,
          row.getInt(4).toDouble, row.getInt(5).toDouble, row.getDouble(6)
        )++ sexFeature
      )
      // 返回标签向量
      LabeledPoint(label, features)
    })

    val Array(trainRDD, testRDD) = titanicRDD.randomSplit(Array(0.8, 0.2))

    /**
     * TODO：c.  使用二分类算法训练模型：SVM、LR、DT和RF、GBT
     */
    // TODO: c.1. 支持向量机

    val svmModel: SVMModel = SVMWithSGD.train(trainRDD, 100)
    val svmPredictionAndLabels: RDD[(Double, Double)] = testRDD.map {
      case LabeledPoint(label, features) => (svmModel.predict(features), label)
    }
    val svmMetrics = new BinaryClassificationMetrics(svmPredictionAndLabels)
    println(s"使用SVM预测评估ROC: ${svmMetrics.areaUnderROC()}")

    // TODO: c.2. 逻辑回归
    val lrModel: LogisticRegressionModel = LogisticRegressionWithSGD.train(trainRDD, 100)
    val lrPredictionAndLabels: RDD[(Double, Double)] = testRDD.map {
      case LabeledPoint(label, features) => (lrModel.predict(features), label)
    }
    val lrMetrics = new BinaryClassificationMetrics(lrPredictionAndLabels)
    println(s"使用LogisticRegression预测评估ROC: ${lrMetrics.areaUnderROC()}")

    // TODO: c.3. 决策树分类
    val dtcModel = DecisionTree.trainClassifier(
      trainRDD, 2, Map[Int, Int](), "gini", 5, 8
    )
    val dtcPredictionAndLabels: RDD[(Double, Double)] = testRDD.map {
      case LabeledPoint(label, features) => (dtcModel.predict(features), label)
    }
    val dtcMetrics = new BinaryClassificationMetrics(dtcPredictionAndLabels)
    println(s"使用DecisionTree预测评估ROC: ${dtcMetrics.areaUnderROC()}")

    // TODO: c.4. 随机森林分类
    val rfcModel = RandomForest.trainClassifier(
      trainRDD, 2, Map[Int, Int](), 10, "sqrt", "gini", 5, 8
    )
    val rfcPredictionAndLabels: RDD[(Double, Double)] = testRDD.map {
      case LabeledPoint(label, features) => (rfcModel.predict(features), label)
    }
    val rfcMetrics = new BinaryClassificationMetrics(rfcPredictionAndLabels)
    println(s"使用RandomForest预测评估ROC: ${rfcMetrics.areaUnderROC()}")


    // TODO: c.5. GBT分类(梯度提升集成学习算法训练模型和预测）
    val gbtModel = GradientBoostedTrees.train(
      trainRDD,
      BoostingStrategy(
        new Strategy(Algo.Classification, Gini, 5, 2, 8),
        SquaredError
      )
    )

    val gbtPredictionAndLabels: RDD[(Double, Double)] = testRDD.map {
      case LabeledPoint(label, features) => (gbtModel.predict(features), label)
    }
    val gbtMetrics = new BinaryClassificationMetrics(gbtPredictionAndLabels)
    println(s"使用GradientBoostedTrees预测评估ROC: ${gbtMetrics.areaUnderROC()}")


    // 程序休眠，为了方便WEB UI监控
    Thread.sleep(1000000)

    // 关闭资源
    spark.stop()

  }

}
