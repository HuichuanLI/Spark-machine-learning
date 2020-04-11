package Rules

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.mllib.fpm.{FPGrowth, FPGrowthModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 * Author : lihuichuan
 * Time   : 2020/4/11
 **/
object FPGrowthDemo {
  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)

    // TODO: 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .appName("FPGrowthDemo").master("local[*]")
      .getOrCreate()

    // 获取SparkContext实例对象
    val sc: SparkContext = spark.sparkContext

    sc.setLogLevel("WARN")

    // TODO: a. 读取样例数据
    val datasRDD: RDD[String] = sc.textFile(s"file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/Rules/sample_fpgrowth.txt",
      minPartitions = 6)

    datasRDD.foreach(println)

    // 将每行数据分割
    val transactionsRDD: RDD[Array[String]] = datasRDD.mapPartitions(iter => {
      iter.map(line => line.split("\\s+"))
    })
    transactionsRDD.cache().count()


    // TODO: b. 使用FP-Growth 算法训练模型（找到频繁项集）
    /**
     * 默认的值： def this() = this(0.3, -1)
     */
    val fpg = new FPGrowth()
      // 设置 最小支持度
      .setMinSupport(0.5)
      // 设置分区数，用于并行计算
      .setNumPartitions(4)

    // 针对训练数据，训练模型
    /**
     * def run[Item: ClassTag](data: RDD[Array[Item]]): FPGrowthModel[Item]
     */
    val fpgModel: FPGrowthModel[String] = fpg.run(transactionsRDD)


    // TODO: c. 查看所有的频繁项集，并且列出它出现的次数
    /**
     * class FPGrowthModel[Item: ClassTag](val freqItemsets: RDD[FreqItemset[Item]])
     */
    val freqItemsetsRDD: RDD[FreqItemset[String]] = fpgModel.freqItemsets
    println(s"Number of frequent itemsets: ${freqItemsetsRDD.count()}")

    val array = freqItemsetsRDD.collect()

    Thread.sleep(10000000)
    // 管理资源
    spark.stop()
  }
}
