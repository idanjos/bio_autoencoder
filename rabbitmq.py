import pika



def send(ip,id, msg):
    connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=ip))
    channel = connection.channel()

    channel.queue_declare(queue=id)

    channel.basic_publish(exchange='', routing_key=id, body=msg)
    print(" [x] Sent Message of " + str(len(msg)) + " characters")
    connection.close()