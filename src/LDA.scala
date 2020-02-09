import kmeans.{assembler, data}
import naive_bayes.spark
import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

import scala.util.Random

/**
 * Author : lihuichuan
 * Time   : 2020/2/9
 **/
object LDA extends App {

  val conf = new SparkConf().setMaster("local").setAppName("iris")
  val spark = SparkSession.builder().config(conf).getOrCreate()

  spark.sparkContext.setLogLevel("WARN") ///日志级别


  val file = spark.read.format("csv").load("/Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/data/iris.data")
  file.show()

  import spark.implicits._

  val random = new Random()
  val data = file.map(row => {
    val label = row.getString(4) match {
      case "Iris-setosa" => 0
      case "Iris-versicolor" => 1
      case "Iris-virginica" => 2
    }

    (row.getString(0).toDouble,
      row.getString(1).toDouble,
      row.getString(2).toDouble,
      row.getString(3).toDouble,
      label,
      random.nextDouble())
  }).toDF("_c0", "_c1", "_c2", "_c3", "label", "rand").sort("rand")
  val assembler = new VectorAssembler()
    .setInputCols(Array("_c0", "_c1", "_c2", "_c3"))
    .setOutputCol("features")

  val dataset = assembler.transform(data)
  val Array(train, test) = dataset.randomSplit(Array(0.8, 0.2))
  train.show()


  val lda = new LDA().setFeaturesCol("features").setK(3).setMaxIter(40)
  val model = lda.fit(train)
  val prediction = model.transform(train)
  //prediction.show()

  // 最大似然估计，最小二乘估计，贝叶斯估计
  val ll = model.logLikelihood(train)
  val lp = model.logPerplexity(train)
  val topics = model.describeTopics(3)
  prediction.select("label", "topicDistribution").show(false)
  println("The topics described by their top-weighted terms:")
  topics.show(false)
  println(s"The lower bound on the log likelihood of the entire corpus: $ll")
  println(s"The upper bound on perplexity: $lp")

}
