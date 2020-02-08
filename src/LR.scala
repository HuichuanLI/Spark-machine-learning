import org.apache.spark.sql.{Encoder, Encoders, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

import scala.util.Random


/**
 * Author : lihuichuan
 * Time   : 2020/2/8
 **/

object LR extends App {


  val conf = new SparkConf().setMaster("local").setAppName("lr")
  val sc = new SparkContext(conf)
  val spark = SparkSession.builder().getOrCreate()

  import spark.implicits._


  val file = spark.read.format("csv").option("header", "true").option("sep", ";").load("/Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/data/house.csv")


  val rand = new Random()
  val data = file.select("square", "price").map(
    row => (row.getAs[String](0).toDouble, row.getString(1).toDouble, rand.nextDouble()))
    .toDF("square", "price", "rand").sort("rand") //强制类型转换过程
  data.show()

  val ass = new VectorAssembler().setInputCols(Array("square")).setOutputCol("features")
  val dataset = ass.transform(data) //特征包装

  val Array(train, test) = dataset.randomSplit(Array(0.8, 0.2)) //拆分成训练数据集和测试数据集


  val lr = new LinearRegression().setStandardization(true).setMaxIter(10)
    .setFeaturesCol("features")
    .setLabelCol("price")
  //创建一个对象
  val model = lr.fit(train) //训练

  model.transform(test).show()


}
