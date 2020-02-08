/**
 * Author : lihuichuan
 * Time   : 2020/2/8
 **/

import org.apache.spark.mllib.linalg._
import breeze.linalg.DenseVector

object Vector extends App {

  val v1 = Vectors.dense(1, 2, 3, 4)
  println(v1)

  val v2 = DenseVector(1, 2, 3, 4)
  //  println(v2)

  println(v2 + v2)
  println(v2.t)
  println(v2 * v2.t)


}
