import org.apache.spark.SparkContext

/**
 * Author : lihuichuan
 * Time   : 2020/2/8
 **/

import org.apache.spark.mllib.{linalg, stat}

//该特征与结果不相关。
//因此，当p值比较小时，证明该假设出现的概率很低，从而有更大的把握拒绝假设，故而我们得出了：
object chi2_test extends App {

  var sc = new SparkContext("local", "chi2_test")
  val data = linalg.Matrices.dense(2, 2, Array(129, 19, 147, 10))
  println(data)

  println(stat.Statistics.chiSqTest(data))


}
