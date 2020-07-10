# #!/usr/bin/python
# # -*- coding:utf-8 -*-

# from pykafka import KafkaClient

# client = KafkaClient(hosts ="192.168.1.140:9092,192.168.1.141:9092,192.168.1.142:9092") #可接受多个client
# #查看所有的topic
# client.topics
# print (client.topics)


# topic = client.topics['test_kafka_topic']#选择一个topic

# message ="test message test message"
# #当有了topic之后呢，可以创建一个producer,来发消息，生产kafka数据,通过字符串形式，
# with topic.get_sync_producer() as producer:
#     producer.produce(message)
# #The example above would produce to kafka synchronously -
# #the call only returns after we have confirmation that the message made it to the cluster.
# #以上的例子将产生kafka同步消息，这个调用仅仅在我们已经确认消息已经发送到集群之后

# #但生产环境，为了达到高吞吐量，要采用异步的方式，通过delivery_reports =True来启用队列接口；
# with topic.get_sync_producer() as producer:
#     producer.produce('test message',partition_key='{}')
# producer=topic.get_producer()
# producer.produce(message)
# print (message)
# import logging as log
# log.basicConfig(level=log.DEBUG)
# from pykafka import KafkaClient
# client = KafkaClient(hosts="ip:5939")
# print (client.topics)
from kafka import KafkaProducer

# 创建生产者
producer = KafkaProducer(bootstrap_servers='localhost:9092')

producer.send('test', b'1100.pdf')
producer.flush()
# producer.send('test',b'1046.pdf')
# producer.flush()
# producer.send('test',b'2.pdf')
# producer.flush()

# producer.send('test',b'1050.pdf')
# producer.flush()
# producer.send('test',b'1100.pdf')
# producer.flush()
# import time
# from kafka import KafkaProducer

# producer=KafkaProducer(bootstrap_servers="localhost:9092")
# i=0
# while True:
#     ts=int(time.time()*1000)
#     producer.send(topic='test',value=str(i),key=str(i),timestamp_ms=ts)
#     producer.flush()
#     print(i)
#     i+=1
#     time.sleep(1)