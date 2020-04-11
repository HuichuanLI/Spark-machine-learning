package Rules

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.fpm.{PrefixSpan, PrefixSpanModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 * 频繁序列挖掘算法PrefixSpan基本使用Demo
 */
object PrefixSpanDemo {

  def main(args: Array[String]): Unit = {

    Logger.getRootLogger.setLevel(Level.WARN)

    // TODO: 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .appName("PrefixSpanDemo").master("local[*]")
      .getOrCreate()

    // 获取SparkContext实例对象
    val sc: SparkContext = spark.sparkContext


    // TODO：构建数据集
    /*
        <a(abc)(ac)d(cf)>
        <(ad)c(bc)(ae)>
        <(ef)(ab)(df)cb>
        <eg(af)cbc>c
        数值替换：
        <1(123)(13)4(36)>
        <(14)3(23)(15)>
        <(56)(12)(46)32>
        <57(16)323>
     */
    val sequencesRDD: RDD[Array[Array[Int]]] = sc.parallelize(
      Seq(
        Array(Array(1), Array(1, 2, 3), Array(1, 3), Array(4), Array(3, 6)), //
        Array(Array(1, 4), Array(3), Array(2, 3), Array(1, 5)), //
        Array(Array(5, 6), Array(1, 2), Array(4, 6), Array(3), Array(2)), //
        Array(Array(5), Array(7), Array(1, 6), Array(3), Array(2), Array(3)) //
      ), //
      numSlices = 2 // 设置RDD分区数目
    ).cache() // 缓存数据


    // TODO: 构建PrefixSpan实例对象，设置最小置信度support
    val prefixSpan: PrefixSpan = new PrefixSpan() // def this() = this(0.1, 10, 32000000L)
      .setMinSupport(0.5) // 设置最小置信度
      .setMaxPatternLength(5) // 设置最大的模式序列长度


    // TODO：使用数据训练获取模型
    val prefixSpanModel: PrefixSpanModel[Int] = prefixSpan.run(sequencesRDD)

    // 获取频繁序列
    /*
      class FreqSequence[Item] @Since("1.5.0") (
        @Since("1.5.0") val sequence: Array[Array[Item]],
        @Since("1.5.0") val freq: Long)
     */
    val freqSequencesRDD: RDD[PrefixSpan.FreqSequence[Int]] = prefixSpanModel.freqSequences

    freqSequencesRDD.collect().foreach(freqSequence => {
      println(
        freqSequence.sequence.map(_.mkString("(", "|", ")")).mkString("<", ", ", ">") +
          " -> " + freqSequence.freq
      )
    })

    println("------------- 华丽的分割线 -------------")
    // TODO: 依据业务，假设 frequency频率大于等于3，模式序列长度等于 2
    freqSequencesRDD
      .filter(freqSequence => {
        freqSequence.freq >= 3 && freqSequence.sequence.length >= 2
      })
      .collect().foreach(freqSequence => {
      println(
        freqSequence.sequence.map(_.mkString("(", "|", ")")).mkString("<", ", ", ">") +
          " -> " + freqSequence.freq
      )
    })

    // TODO: 保存模型
    //    prefixSpanModel.save(sc, "")

    // 往往保存的是 结果数据，并不是模型
    //    freqSequencesRDD.saveAsTextFile("")

    Thread.sleep(10000000)
    // 管理资源
    spark.stop()
  }

}
