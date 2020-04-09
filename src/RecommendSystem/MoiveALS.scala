package RecommendSystem


import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Author : lihuichuan
 * Time   : 2020/4/9
 **/
object MoiveALS {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().
      setAppName("moiveAlS").setMaster("local[*]")
    //      .set("spark.driver.extraJavaOptions", "-Xss10m")

    val sc = SparkContext.getOrCreate(sparkConf)

    sc.setLogLevel("WARN") ///日志级别

    // TODO: 2. 读取 电影评分数据
    val RawRating = sc.textFile("file:///Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/RecommendSystem/movielens/ml-100k/u.data")


    val ratingsRDD: RDD[Rating] = RawRating
      .filter(line => line.length > 0 && line.split("\t").length == 4)
      .map(line => {
        val Array(userId, moiveId, rating, _) = line.split("\t")
        Rating(userId.toInt, moiveId.toInt, rating.toDouble)

      })

    // TODO： 4. 调用ALS算法中显示训练函数训练模型
    import org.apache.spark.mllib.recommendation.ALS
    // 迭代次数为20，特征数为10
    val alsModel: MatrixFactorizationModel = ALS.train(ratings = ratingsRDD, rank = 10, iterations = 20)

    // TODO: 模型评估
    import org.apache.spark.mllib.evaluation.RegressionMetrics


    val uprsRDD: RDD[((Int, Int), Double)] = ratingsRDD.map(tuple => ((tuple.user, tuple.product), tuple.rating))
    val predictUprs: RDD[((Int, Int), Double)] = alsModel
      .predict(uprsRDD.map(_._1))
      .map(tuple => ((tuple.user, tuple.product), tuple.rating))
    val predictAndArtual: RDD[((Int, Int), (Double, Double))] = predictUprs.join(uprsRDD)
    val metrics = new RegressionMetrics(predictAndArtual.map(_._2))

    println(s"RMSE = ${metrics.rootMeanSquaredError}")
    println(s"MSE = ${metrics.meanSquaredError}")

    /**
     * 获取模型MatrixFactorizationModel就是里面包含两个矩阵：
     * -a. 用户因子矩阵
     *         alsModel.userFeatures
     * -b. 产品因子矩阵
     *
     */
    // userId -> Features
    val userFeatures: RDD[(Int, Array[Double])] = alsModel.userFeatures
    userFeatures.take(10).foreach(tuple => println(tuple._1 + " -> \n\t" + tuple._2.mkString(",")))
    // productId -> Features
    val productFeatures: RDD[(Int, Array[Double])] = alsModel.productFeatures
    productFeatures.take(10).foreach(tuple => println(tuple._1 + " -> \n\t" + tuple._2.mkString(",")))

    // TODO 5. 推荐与预测评分
    // a. 预测某个用户对某个产品的评分  def predict(user: Int, product: Int): Double
    val predictRating: Double = alsModel.predict(196, 242)
    println(s"预测用户196对电影242的评分：$predictRating")

    println("----------------------------------------")

    // b. 为某个用户推荐十部电影  def recommendProducts(user: Int, num: Int): Array[Rating]
    val rmdMovies: Array[Rating] = alsModel.recommendProducts(196, 10)
    rmdMovies.foreach(println)

    println("----------------------------------------")

    // c. 为某个电影推荐10个用户  def recommendUsers(product: Int, num: Int): Array[Rating]
    val rmdUsers = alsModel.recommendUsers(242, 10)
    rmdUsers.foreach(println)
    // TODO: 6. 将训练得到的模型进行保存，以便后期加载使用进行推荐
    alsModel.save(sc, "/Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/RecommendSystem/als/ml-als-model")

    // TODO: 7. 从文件系统中记载保存的模型，用于推荐预测
    // override def load(sc: SparkContext, path: String): MatrixFactorizationModel
    val loadAlsModel: MatrixFactorizationModel = MatrixFactorizationModel
      .load(sc, "/Users/hui/Desktop/Hadoop/spark-machine-learning/Spark-machine-learning/src/RecommendSystem/als/ml-als-model")
    // 使用加载预测
    val loaPredictRating: Double = loadAlsModel.predict(196, 242)
    println(s"加载模型预测用户196对电影242的评分：$loaPredictRating")
    Thread.sleep(1000000)
    sc.stop()
  }
}
