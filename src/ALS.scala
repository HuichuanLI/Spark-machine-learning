import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.sql.SparkSession

/**
 * Author : lihuichuan
 * Time   : 2020/2/9
 **/
object ALS extends App {
  val conf = new SparkConf().setMaster("local").setAppName("RS")
  val spark = SparkSession.builder().config(conf).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val parseRating = (string: String) => {
    val stringArray = string.split("\t")
    Rating(stringArray(0).toInt, stringArray(1).toInt, stringArray(2).toFloat)
  }

  import spark.implicits._

  val data = spark.read.textFile("./data/u.data")
    .map(parseRating)
    .toDF("userID", "itemID", "rating")
  val Array(traing, test) = data.randomSplit(Array(0.8, 0.2))

  val als = new ALS()
    .setMaxIter(20)
    .setUserCol("userID")
    .setItemCol("itemID")
    .setRatingCol("rating")
    .setRegParam(0.01) //正则化参数

  val model = als.fit(traing)
  model.setColdStartStrategy("drop") //冷启动策略，这是推荐系统的一个重点内容哦～

  val predictions = model.transform(test)
  //predictions.show(false)//根据(userID,itemID)预测rating


  //MovieLens
  val users = spark.createDataset(Array(196)).toDF("userID")
  //users.show(false)
  model.recommendForUserSubset(users, 10).show(false) //想一想工业实践该怎么结合这段代码？

  //模型评估
  val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol("rating")
    .setPredictionCol("prediction")
  val rmse = evaluator.evaluate(predictions)
  println(s"Root-mean-square error is $rmse \n")

}
