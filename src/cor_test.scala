import org.apache.spark.SparkContext

/**
 * Author : lihuichuan
 * Time   : 2020/2/8
 **/
import org.apache.spark.mllib.stat._

object cor_test extends App {
  var sc = new SparkContext("local", "analyse")
  var txt = sc.textFile("/Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/data/beijing_cor.txt")
  val data = txt.flatMap(_.split(",")).map(_.toDouble)
  val year = data.filter(_ > 1000)
  val values = data.filter(_ <= 1000)
  // 相关系数
  println(Statistics.corr(year, values))
}
