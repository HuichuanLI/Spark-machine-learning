import org.apache.spark.SparkContext

/**
 * Author : lihuichuan
 * Time   : 2020/2/8
 **/
import org.apache.spark.mllib.{stat, linalg}

object dataAnalyse extends App {
  var sc = new SparkContext("local", "analyse")
  var txt = sc.textFile("/Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/data/beijing.txt")
  // foreach 打印
  // 查看数据
  val array = txt.flatMap(_.split(",")).take(10)
  for (elem <- array) {
    println(elem)
  }

  // 每个都是RDD
  val data = txt.flatMap(_.split(",")).map(value => linalg.Vectors.dense(value.toDouble))

  val res = stat.Statistics.colStats(data)
  println(res.max)
  println(res.min)
  println(res.mean)
  println(res.variance)


  //  println(txt.count())
}
