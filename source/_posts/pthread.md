---
title: Linux下的多线程编程:生产消费
date: 2022-03-13 22:15:26
tags: [linux]
---
**线程互斥**，pthread_mutex_t 线程互斥锁，全局变量；pthread_mutex_init 互斥锁初始化；pthread_mutex_create 分别创建生产者和消费者线程；**pthread_join** 线程阻塞执行函数；pthread_mutex_destroy 线程互斥锁销毁，在所有PV量也销毁之后。

**线程信号量**，sem_t datasem，blanksem两个信号量（可用数据信号量，空闲资源信号量）；sem_init()（**datasem 初始为0，blanksem初始为最大数量NUM**）sem为指向信号量结构的一个指针；**pshared不为０时此信号量在进程间共享，否则只能为当前进程的所有线程共享**；value给出了信号量的初始值。***生产者***：首先sem_wait(&blanksem) P操作，等待申请空闲资源，再上线程互斥锁，接着进行操作，完成之后再sem_post(&datasem) V操作，通知增加了数据资源，最后线程解锁。***消费者***：首先sem_wait(&datasem) P操作，等待申请数据资源，再上线程互斥锁，接着进行操作，完成之后再sem_post(&blanksem) V操作，通知增加了空间资源，最后线程解锁。等待所有线程结束后，收回资源，sem_destroy销毁信号量。

**线程执行**，P操作（ -1，生产者：申请减少空闲资源；消费者：申请减少可用数据资源），pthread_mutex_lock 线程互斥锁，进入公共区域，V操作（+1，生产者：通知增加可用数据资源；消费者：通知增加空闲资源），pthread_mutex_unlock线程互斥解锁。

**线程**，pthread_t 定义线程，创建线程pthread_create（传入线程地址pthread_t，线程属性pthread_attr_t 默认初始NULL属性，线程函数，线程函数输入变量arg<指定为线程的编号>），pthread_join放入线程，阻塞函数，先让生产者线程生产，再放入消费者；通过阻塞，让线程保持生产消费状态，等到所有生产消费者线程结束为止，才收回线程资源。
