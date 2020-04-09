package RecommendSystem

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Author : lihuichuan
 * Time   : 2020/4/9
 **/
object MusicALS {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().
      setAppName("musicAlS").setMaster("local[*]")
    val sc = SparkContext.getOrCreate(sparkConf)

    sc.setLogLevel("WARN") ///日志级别


    val rawUserArticleText = sc.textFile("file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/RecommendSystem/lastfm/user_artist_data.txt", minPartitions = 6)

    println(rawUserArticleText.count())

    println(rawUserArticleText.first())

    println(rawUserArticleText.map(_.split("\\s")(0)).distinct().count())

    println(rawUserArticleText.map(_.split("\\s")(1)).distinct().count())

    // TODO b. 读取artist_alias.txt 数据， 艺术家别名数据
    val rawArtistAliasRDD = sc.textFile("file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/RecommendSystem/lastfm/artist_alias.txt")
    val artistAliasMap: collection.Map[Int, Int] = rawArtistAliasRDD.flatMap(line => {
      val tokens = line.split("\t")
      // 由于某些原因有些艺术家只有一个ID
      if (tokens(0).isEmpty) {
        None
      } else {
        Some((tokens(0).toInt, tokens(1).toInt))
      }
    }).collectAsMap()

    // 广播变量进行广播Map集合
    val broadcastArtistAlias = sc.broadcast(artistAliasMap)

    // TODO c. 获取ALS算法所需的数据集 RDD[Rating]
    val trainingRDD: RDD[Rating] = rawUserArticleText.map(line => {
      // 用户-艺术家-次数，字符串分割
      val Array(userID, artistID, count) = line.split("\\s").map(_.toInt)

      // 将艺术家ID进行统一化
      val finalArtistID = broadcastArtistAlias.value.getOrElse(artistID, artistID)

      // 返回Rating对象
      Rating(userID, finalArtistID, count.toDouble)
    })

    // 将训练数据集RDD进行缓存，因为训练模型多次训练模型，数据集使用多次
    trainingRDD.cache()


    // TODO d. 使用ALS训练模型（基于训练数据集）
    // def train(ratings: RDD[Rating], rank: Int, iterations: Int, lambda: Double)
    val model: MatrixFactorizationModel = ALS.trainImplicit(trainingRDD, 10, 10, 0.01, 1.0)

    // 用户特征矩阵
    println(model.userFeatures.first()._2.mkString(", "))
    // 产品特征矩阵
    println(model.productFeatures.first()._2.mkString(", "))

    /**
     * 定义函数，评估ALS模型，评估指标为RMSE
     */
    def alsModelEvaluate(alsModel: MatrixFactorizationModel, rdd: RDD[Rating]): Double = {
      // 获取每个用户及对应艺术家ID
      val usersProductRDD: RDD[(Int, Int)] = rdd.map(rating => (rating.user, rating.product))
      // def predict(usersProducts: RDD[(Int, Int)]): RDD[Rating]
      val predictRDD: RDD[Rating] = alsModel.predict(usersProductRDD)
      //
      val predictionAndActualRDD = predictRDD
        .map(r => ((r.user, r.product), r.rating))
        .join(rdd.map(r => ((r.user, r.product), r.rating)))
        .map {
          case ((userId, productId), (predict, actual)) => (predict, actual)
        }
      //  def this(predictionAndObservations: RDD[(Double, Double)])
      val metrics = new RegressionMetrics(predictionAndActualRDD)

      metrics.rootMeanSquaredError
    }

    val evaluations: Array[(Double, MatrixFactorizationModel, Int, Int, Double)] = for {
      rank <- Array(10, 50)
      iterations <- Array(10, 20)
      lambda <- Array(0.1, 0.01)
    } yield {
      val als_model = ALS.trainImplicit(trainingRDD, rank, iterations, lambda, 1.0)
      // 评估模型，计算RMSE
      (alsModelEvaluate(als_model, trainingRDD), model, rank, iterations, lambda)
    }

    // 找到最佳模型，依据 rmse 升序排序，获取最小值的模型
    val sortEvaluations = evaluations.sortBy(_._1)
    sortEvaluations.foreach(println)

    val bestModel = sortEvaluations(0)._2
    println(bestModel)

    Thread.sleep(10000)
    sc.stop()

  }
}
