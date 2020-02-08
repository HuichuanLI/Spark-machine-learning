import org.apache.spark.SparkContext

/**
 * Author : lihuichuan
 * Time   : 2020/2/8
 **/
object WordCount {
  def main(args: Array[String]): Unit = {
    var sc = new SparkContext("local", "wordcount")
    var file = sc.textFile("/Users/hui/Desktop/bigdata/spark-2.4.4-bin-hadoop2.7/LICENSE")
    file.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_ + _).sortBy(-_._2).foreach(println)

  }
}
