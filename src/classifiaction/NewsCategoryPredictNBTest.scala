package classifiaction

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.{HashingTF, IDF, IDFModel, Word2Vec, Word2VecModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

/**
 * Author : lihuichuan
 * Time   : 2020/4/10
 **/
object NewsCategoryPredictNBTest {
  def main(args: Array[String]): Unit = {

    // a. 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .appName("NewsCategoryPredictNBTest")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    // 获取SparkContext上下文实例对象
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    // b. 读取制表符分割cvs格式数据，通过自定义Schema信息
    // b.1 定义schema  ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP

    val schema = StructType(
      Array(
        StructField("id", IntegerType, nullable = true),
        StructField("title", StringType, nullable = true),
        StructField("url", StringType, nullable = true),
        StructField("publisher", StringType, nullable = true),
        StructField("category", StringType, nullable = true),
        StructField("story", StringType, nullable = true),
        StructField("hostname", StringType, nullable = true),
        StructField("timestamp", StringType, nullable = true)
      )
    )

    // b.2 读取数据
    val newsDF: DataFrame = spark.read
      .option("sep", "\t")
      .schema(schema)
      .csv("file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/classifiaction/newsdata/newsCorpora.csv")
    //
    newsDF.printSchema()
    newsDF.show(20, truncate = true)

    // TODO: 统计各个类别的数据，看看类别数据中是否出现数据倾斜
    newsDF.select($"category").groupBy($"category").count().show(10, truncate = false)

    // b.3 提取字段数据
    val titleCategoryRDD: RDD[(Double, String)] = newsDF.select($"title", $"category").rdd.map(row => {
      // 获取类别数据
      // b = business, t = science and technology, e = entertainment, m = health
      val category = row.getString(1) match {
        case "b" => 0.0 // b = business
        case "t" => 1.0 // t = science and technology
        case "e" => 2.0 // e = entertainment
        case "m" => 3.0 // m = health
        case _ => -1
      }
      // 以二元组形式返回
      (category, row.getString(0))
    })
    // b.4 从 新闻标题 title 提取特征值（转换为数值类型）
    /**
     * 使用次贷模型（BOW）提取文本特征数据
     * TF，IDF
     */
    // i. 创建HashingTF，指定单词数量 为100000, 默认值为2的20次方=1048576
    val hashTF = new HashingTF(100000)
    // 特征转换
    val lpsRDD: RDD[LabeledPoint] = titleCategoryRDD.map {
      case (category, newsTitle) =>
        // 对 新闻标题 进行分词
        val words = newsTitle.split("\\s+")
          // 将所有单词转换为小写字母
          .map(word => word.toLowerCase.replace(",", "").
            replace(".", "")).toSeq

        val tf: linalg.Vector = hashTF.transform(words)
        // 返回
        LabeledPoint(category, tf)
    }.filter(lp => lp.label != -1)
    lpsRDD.take(5).foreach(println)

    // 获取IDF模型
    val tfRDD: RDD[linalg.Vector] = lpsRDD.map(_.features)

    // 构建一个IDF： 通过TF值获取
    val idfModle: IDFModel = new IDF().fit(tfRDD)

    // 给 TF 进行加权
    val lpsTfIdfRDD = lpsRDD.map {
      case LabeledPoint(category, tf) => LabeledPoint(category, idfModle.transform(tf))
    }

    lpsTfIdfRDD.take(5).foreach(println)

    // TODO：划分数据集为训练数据集和测试数据集
    val Array(trainingRDD, testingRDD) = lpsTfIdfRDD.randomSplit(Array(0.8, 0.2))


    // c. 使用朴素贝叶斯算法训练模型
    val nbModel: NaiveBayesModel = NaiveBayes.train(trainingRDD)

    // 使用测试数据集进行预测
    val nbPredictAndActualRDD = testingRDD.map {
      case LabeledPoint(label, features) => (nbModel.predict(features), label)
    }
    nbPredictAndActualRDD.take(10).foreach(println)

    // 多分类评估
    val metrics = new MulticlassMetrics(nbPredictAndActualRDD)
    println(s"使用朴素贝叶斯模型预测新闻分类：精确度ACC = ${metrics.accuracy}")
    // 混淆矩阵
    println(metrics.confusionMatrix)

    // d. 模型持久化（将模型保存）
    //    nbModel.save(sc, "datas/news-nb-model" + System.currentTimeMillis())

    // 加载模型并进行预测 , 实际值为 b
    val title = "Fed official says weak data caused by weather, should not slow taper"
    // 特征提取
    val features = idfModle.transform(
      hashTF.transform(
        title.split("\\s+").map(_.toLowerCase.replace(",", "").replace(".", ""))
      )
    )
    // load model from local fs
    val predictCategory: String = nbModel
    .predict(features) match {
      case 0 => "b"
      case 1 => "t"
      case 2 => "e"
      case 3 => "m"
      case _ => "unknown"
    }
    println(s"Predict Category: $predictCategory")

    println("=======================================================")

    // TODO: Word2Vec: 将单词表示成一个向量，可以用于计算两个单词之间的相似度
    // http://spark.apache.org/docs/2.2.0/mllib-feature-extraction.html#word2vec

    // 将文本分割单词
    val inputRDD: RDD[Seq[String]] = titleCategoryRDD.map{
      case (category, newsTitle) =>
        newsTitle
          .split("\\s+")
          .map(word => word.trim.toLowerCase.replace(",", "").replace(".", ""))
          .toSeq
    }
    // 创建Word2Vec实例对象
    val word2Vec = new Word2Vec()
    // 使用数据训练模型
    val word2VecModel: Word2VecModel = word2Vec.fit(inputRDD)

    // 使用模型找出某个单词 前20 个 相近词汇
    val sysnonys: Array[(String, Double)] = word2VecModel.findSynonyms("STOCK".toLowerCase, 10)
    for((word, cosineSimilarity) <- sysnonys){
      println(s"$word -> $cosineSimilarity")
    }





    // 线程休眠
    Thread.sleep(1000000)
    // 关闭资源
    spark.stop()
  }
}
