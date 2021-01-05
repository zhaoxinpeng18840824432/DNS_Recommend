package com.zxp

import breeze.linalg.rank
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.jblas.DoubleMatrix

import scala.io.StdIn
object OfflineRecommend {

  /**
   * Rating数据集，用户对于域名的访问次数 数据集，用，分割
   *
   * 1,           用户的ID
   * 16,          域名的ID
   * 50,         用户对于域名的访问次数
   */
  case class DNSRating(user: String, qname: String, allnum: BigDecimal)
  // 用户 id
  case class UserId(uid: Int, user:String)
  //DNS id
  case class QnameId(uid: Int, qname:String)
  //推荐
  case class Recommendation(rid: Int, r: Double)
  // 用户的推荐
  case class UserRecs(uid: Int, recs: Seq[Recommendation])
  val MONGODB_RATING_COLLECTION = "Rating"
  val USER_MAX_RECOMMENDATION = 10
  val USER_RECS = "UserRecs"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]"
    )

    //创建一个SparkConf配置
    val sparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster(config("spark.cores")).set("spark.executor.memory", "6G").set("spark.driver.memory", "2G")
    //基于SparkConf创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    //创建一个MongoDBConfig

    import spark.implicits._

    /**
     * 计算用户推荐矩阵，利用协同过滤里的ALS算法计算用户推荐矩阵
     */
    //抽取评分数据集
    /*
    select qname,sum(num) as allnum,d_ip,d_port from(
     select qname,count(qname) as num,d_ip,d_port from t_dams_dnsc2r where s_ip='192.168.133.64' group by d_ip,d_port,qname
     union all
     select qname,count(qname) as num,d_ip,d_port from t_dams_dnsr2a where s_ip='192.168.133.64' group by d_ip,d_port,qname
     union all
     select qname,count(qname) as num,d_ip,d_port from t_dams_dnsc2f where s_ip='192.168.133.64' group by d_ip,d_port,qname) s
     group by d_ip,d_port,qname
     */

    //链接postgresql数据库
    val jdbc= spark.read.format("jdbc").options(

      Map("url"->"jdbc:postgresql://12.12.12.8:5431/dns",

        "driver"->"org.postgresql.Driver",

        "user"-> "dbuser",

       /* "numPartitions" -> "4",

        "lowerBound" -> "1",*/

        "fetchSize"->"100",

        "fetchSize"->"100",

        "password"->"1"))

    //查询sql：用户user，域名qname，访问次数allnum
    //获得用户-域名-访问次数  矩阵
    val ratingMatrixDS=jdbc.option("dbtable","(select d_ip||':'||d_port  as user,qname,sum(num) as allnum from(select qname,count(qname) as num,d_ip,d_port from t_dams_dnsc2r where s_ip='192.168.133.64' group by d_ip,d_port,qname union all select qname,count(qname) as num,d_ip,d_port from t_dams_dnsr2a where s_ip='192.168.133.64' group by d_ip,d_port,qname union all select qname,count(qname) as num,d_ip,d_port from t_dams_dnsc2f where s_ip='192.168.133.64' group by d_ip,d_port,qname) s group by d_ip,d_port,qname) table_s")
      .load().as[DNSRating]

    //分别获得所有用户名和域名
    val userDS=ratingMatrixDS.map(_.user).distinct()
    val qnameDS=ratingMatrixDS.map(_.qname).distinct()

    //val userSchema: StructType = userDS.schema.add(StructField("uid", LongType))
    //val qnameSchema: StructType = qnameDS.schema.add(StructField("qid", LongType))
    // DataDS转RDD 然后调用 zipWithIndex
    //为用户名和域名分配ID
    val userWithIdRdd = userDS.rdd.zipWithIndex()
    val qnameWithIdRdd= qnameDS.rdd.zipWithIndex()

    //单独获取用户名和域名的ID
    val userIdRdd=userWithIdRdd.map(_._2.toInt)
    val qnameIdRdd=qnameWithIdRdd.map(_._2.toInt)

    //数据转换为DataFrame
    val userWithIdDF=userWithIdRdd.map(a=>UserId(a._2.toInt,a._1)).toDF()
    val qnameWithIdDF=qnameWithIdRdd.map(a=>QnameId(a._2.toInt,a._1)).toDF()

    //join方法获得  用户-域名-访问次数-用户ID-域名ID  矩阵
    val ratingMatrixWithIdDS=ratingMatrixDS.join(userWithIdDF,"user").join(qnameWithIdDF,"qname")
    ratingMatrixWithIdDS.limit(200).foreach(println(_))
    // 将添加了索引的RDD 转化为DataFrame
    //创建模型训练所需的数据集，获得  用户ID-域名ID-访问次数  矩阵
    val trainRDD: RDD[Rating] = ratingMatrixWithIdDS.map(rating => Rating(rating.getInt(3), rating.getInt(4), rating.getDecimal(2).doubleValue())).rdd
    trainRDD.take(200).foreach(println(_))
    //这三个参数是怎么确定的呢？其实是我们自己测出来的，
    //找出三个最优的参数的原理其实用到的数 "均方跟误差"，如果均方跟误差越小，那么就说名参数越优。
    val (rank, iterations, lambda) = (50, 10, 0.01)
    //创建训练模型 trainRDD:需要训练的数据集，rank：计算时候矩阵的特征数量，iterations：迭代计算的次数，
    val model: MatrixFactorizationModel = ALS.train(trainRDD, rank, iterations, lambda)
    println("train end")
    //StdIn.readLine()

    //创建一个用户与产品的矩阵：用户ID-域名ID 矩阵
    val userProducts: RDD[(Int,Int)] = userIdRdd.cartesian(qnameIdRdd)
    //利用模型开始计算
    val ratingRDD: RDD[Rating] = model.predict(userProducts)
    ratingRDD.take(400).foreach(println(_))
    println("predict end")
    //StdIn.readLine()
    //获取评分大于100的推荐
     val userRecsRdd=ratingRDD.filter(_.rating > 100)
      .map(rating => (rating.user, (rating.product, rating.rating)))
    userRecsRdd.collect().foreach(println(_))
    println("filter end")
    //StdIn.readLine()

    //对推荐结果进行排序，输出每个用户排序前10的推荐域名
    val userRecsDF: DataFrame =userRecsRdd.groupByKey().map {
      case (user, recs) => UserRecs(user, recs.toList.sortWith(_._2 > _._2).take(10).map(x => Recommendation(x._1, x._2)))
    }.toDF()

    userRecsDF.collect().foreach(println(_))

    spark.stop()
  }
}

