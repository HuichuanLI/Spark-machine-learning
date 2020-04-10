package classifiaction


import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, NaiveBayes}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author : lihuichuan
 * Time   : 2020/4/9
 **/

/**
 * Spark MLlib机器学习算法中，基于RDD实现分类算法基础使用
 * 分类算法，需要数据集格式：
 * def run(input: RDD[LabeledPoint]): Model
 * 标签向量
 * case class LabeledPoint(label: Double, features: Vector)
 *     - 标签label：
 * 预测值，分类算法中就是类别
 *     - features:
 * 向量Vector，分为稀疏向量（SparseVector）和稠密向量（DenseVector）
 *     - 创建稠密向量
 * 工具类Vectors中方法：
 * def dense(values: Array[Double]): Vector = new DenseVector(values)
 *     - 无论标签还是每个特征值必须是Double类型（Python中数值类型，Float）
 *
 * 分类数据集鸢尾花数据集：
 *   - 说明：
 * 本身数据集中有150条数据，三种类别鸢尾花
 *   - 数据样本
 *     5.1,3.5,1.4,0.2,Iris-setosa
 * 每行数据各个字段中，使用逗号进行隔开，共五个字段，最后一个字段：类别，前面四个字段：特征值
 *     - 花萼的长度和宽度
 *       5.1,3.5
 *     - 花瓣的长度和宽度
 *       1.4,0.2
 */
object irisClassification {
  def main(args: Array[String]): Unit = {
    // TODO：构建SparkSession实例对象
    val spark = SparkSession.builder()
      .appName("SparkMLlibClassificationTest")
      .master("local[*]")
      .getOrCreate()

    // 获取SparkContext上下文对象
    val sc = spark.sparkContext
    // 设置日志级别
    sc.setLogLevel("WARN")
    /**
     * TODO：a. 读取鸢尾花数据集
     */
    val irisDF: DataFrame = spark.read.csv("file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/classifiaction/iris.data")
    // 样本数据
    irisDF.show(60, truncate = false)

    /**
     * TODO: b. 特征工程，提取特征值，组合到标签向量LabeledPoint
     */
    // 提取出 类别的值，转换为数值类型，从0开始
    val categoryMap: collection.Map[String, Long] = irisDF.rdd
      // 获取所有类别，并进行去重操作
      .map(row => row.getString(4)).distinct()
      .zipWithIndex().collectAsMap()

    categoryMap.foreach(println)
    // 通过广播变量将类别Map集合广播到Executors中
    val categoryMapBroadcast = sc.broadcast(categoryMap)

    // 提取特征标签数据集
    val irisRDD = irisDF.rdd.map(row => {
      // 创建稠密向量：Double类型数组
      val features = Vectors.dense(
        Array(row.getString(0), row.getString(1), row.getString(2), row.getString(3)).map(_.toDouble)
      )
      // 获取标签
      val label: Double = categoryMapBroadcast.value(row.getString(4)).toDouble
      // 返回标签向量
      LabeledPoint(label, features)
    })

    /**
     * TODO：将数据集分为两部分：
     *1. 第一部分为训练数据集（传入到算法中训练得到模型）；
     *2. 第二部分为测试数据集（传入模型中得到预测值，评估模型性能）
     */
    val Array(trainingRDD, testingRDD) = irisRDD.randomSplit(Array(0.8, 0.2))

    /**
     * TODO： c. 使用多分类算法，针对训练数据集训练模型
     */
    // TODO: c.1. logistic regression


    val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(10).run(trainingRDD)
    var nbScoreAndLabels = testingRDD.map {
      case LabeledPoint(label, features) => (lrModel.predict(features), label)
    }
    nbScoreAndLabels.foreach(println)
    var nbMetrics = new MulticlassMetrics(nbScoreAndLabels)
    println(s"Logestic Regression预测评估ACC：${nbMetrics.accuracy}")






    // TODO: c.2. naive Bayes

    val nbModel = NaiveBayes.train(trainingRDD)
    nbScoreAndLabels = testingRDD.map {
      case LabeledPoint(label, features) => (nbModel.predict(features), label)
    }
    nbScoreAndLabels.foreach(println)
    nbMetrics = new MulticlassMetrics(nbScoreAndLabels)
    println(s"使用NaiveBayes预测评估ACC：${nbMetrics.accuracy}")

    // TODO: c.3. decision trees

    /**
     * def trainClassifier(
     * // 训练数据集
     * input: RDD[LabeledPoint],
     * // 分类的类别数
     * numClasses: Int,
     * // 特征值如果有类别特征，告知其信息
     * categoricalFeaturesInfo: Map[Int, Int],
     * // 不纯度度量方式，分类算法来说：熵 entropy或 基尼系数gini
     * impurity: String,
     * // 构建的树的深度
     * maxDepth: Int,
     * // 节点的分支数，一般值为2的N次方，如为2，此树就是二叉树
     * maxBins: Int
     * ): DecisionTreeModel
     */
    val dtcModel = DecisionTree.trainClassifier(trainingRDD,
      3, Map[Int, Int](), "gini", 5, 8)
    val dtcScoreAndLabels = testingRDD.map {
      case LabeledPoint(label, features) => (dtcModel.predict(features), label)
    }
    dtcScoreAndLabels.foreach(println)
    val dtcMetrics = new MulticlassMetrics(dtcScoreAndLabels)
    println(s"使用DecisionTree预测评估ACC：${dtcMetrics.accuracy}")


    Thread.sleep(100000)
    sc.stop()
  }
}
