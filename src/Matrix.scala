/**
 * Author : lihuichuan
 * Time   : 2020/2/8
 **/

import org.apache.spark.mllib.linalg._
import breeze.linalg._

object Matrix extends App {

  // 默认以列为先
  val m1 = Matrices.dense(2, 3, Array(1, 2, 3, 4, 5, 6))
  println(m1)

  //按照行
  val m2 = breeze.linalg.DenseMatrix(Array(1, 2, 3), Array(4, 5, 6))
  println(m2)


  val m3 = breeze.linalg.DenseMatrix(Array(1, 2, 3, 4, 5, 6))
  println(m3.reshape(2, 3))


  println(m3.reshape(2, 3).t)


  println(m3.reshape(2, 3) * m3.reshape(2, 3).t)

}
