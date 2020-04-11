package Rules

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.fpm.AssociationRules.Rule
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.mllib.fpm.{FPGrowth, FPGrowthModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

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


    freqItemsetsRDD.collect()
      .filter(itemset => itemset.items.length > 1 && itemset.items.length < 4)
      .foreach(itemset => println(itemset))


    /**
     * 依据获取的 频繁项集 生成 规则
     *
     * def generateAssociationRules(confidence: Double): RDD[AssociationRules.Rule[Item]]
     * 置信度：
     * confidence，在X发生的条件下，Y发生的概率，此处为最小的值，由此过滤
     *
     * class Rule[Item] private[fpm] (
     * // 表示 前项， hypotheses of the rule， 规则的假设
     * val antecedent: Array[Item],
     * // 表示 后项， conclusion of the rule， 规则的结论
     * val consequent: Array[Item],
     * freqUnion: Double, // 共同出现的次数
     * freqAntecedent: Double // 前项出现的次数
     * )
     * *
     */

    // 通过模型生成关联规则，设置最小置信度过滤数据
    val rulesRDD: RDD[Rule[String]] = fpgModel.generateAssociationRules(0.8)
    // 查看规则的数量
    println(s"Rules Count = ${rulesRDD.count()}")
    // 查看生成的所有规则
    rulesRDD.collect().foreach(println)

    // TODO：依据关联规则，针对业务，得到推荐列表

    println("==============================")
    val rmdItemsRDD = rulesRDD
      .mapPartitions(iter => {
        iter.map(rule =>
          (rule.antecedent.mkString(","), (rule.consequent.mkString(","), rule.confidence))
        )
      })
      .aggregateByKey(ListBuffer[(String, Double)](), 6)(
        //            // seqOp: (U, V) => U
        (u, v) => {
          u += v
          u
        },
        //            // combOp: (U, U) => U
        (u1, u2) => {
          u1 ++= u2
          u1
        }
      )
    rmdItemsRDD.cache().count()
    rmdItemsRDD.foreach(println)
    //    查看推荐结果
    rmdItemsRDD.coalesce(1).foreachPartition(iter => {
      iter
        // TODO: 当查看某Item时，推荐查看另外Items
        //        .filter(_._1.split(",").length == 1)
        .foreach(item => {
          println("看了此Item[" + item._1 + "]的又看了看:")
          if (item._2.length > 0) {
            println(item._2)
          }
        })
    })

    Thread.sleep(10000000)
    // 管理资源
    spark.stop()
  }
}
