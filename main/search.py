import os 
import zmq 

import time 
import json 

import subprocess 
import multiprocessing as mp 

from os import path 
from utilities.utils import * 
from logger.log import logger 

def finder_process(router_port, publisher_port, process_id, router_readyness, workers_states_queue):
    try:
        ctx = zmq.Context()
        req_socket = ctx.socket(zmq.REQ)
        subscriber_socket = ctx.socket(zmq.SUB)

        req_socket.connect(f'tcp://localhost:{router_port}')
        subscriber_socket.connect(f'tcp://localhost:{publisher_port}')
        
        req_socket.setsockopt_string(zmq.IDENTITY, process_id)
        subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, 'KILL')
        
        req_socket_poller = zmq.Poller()
        subscriber_socket_poller = zmq.Poller()

        req_socket_poller.register(req_socket, zmq.POLLIN)
        subscriber_socket_poller.register(subscriber_socket, zmq.POLLIN)

        logger.info(f'worker : {process_id} is initialized')
        router_readyness.wait()  # wait until thr router is ready...! 
        
        req_socket.send_multipart([b'ready', process_id.encode('utf-8')])
        workers_states_queue.put((process_id, 'alive'))

        keep_processing = True 
        while keep_processing:
            subscriber_events = dict(subscriber_socket_poller.poll(100))
            if subscriber_socket in subscriber_events.keys():
                if subscriber_events[subscriber_socket] == zmq.POLLIN: 
                    topic, msg = subscriber_socket.recv_multipart()
                    if topic.decode() == 'KILL':
                        logger.info(f'worker {process_id} got kill signal')
                        keep_processing = False 
            
            req_events = dict(req_socket_poller.poll(100))
            if req_socket in req_events.keys():
                if req_events[req_socket] == zmq.POLLIN: 
                    inc = req_socket.recv_multipart()
                    sep, incoming_req = inc 
                    json_req = json.loads(incoming_req.decode('utf-8'))
                    logger.info(f'worker : {process_id} get req : {json_req}')
                    source, target = json_req['source'], json_req['target']
                    script = f'python -m searching.search --extractor models/vgg16.h5 --target {target} --input {source}'   
                    logger.info('the next script will start : ')
                    logger.info(script)
                    
                    #time.sleep(1)  # some work ... :) 
                    
                    response = subprocess.run(script, shell=True)
                    if response.returncode == 0:
                        logger.success(f'the similarity was computed for {source}')
                    else:
                        logger.error(f'some erros was triggered during the computation : {source}')
                    
                    logger.info(f'worker {process_id} finish the work')
                    req_socket.send_multipart([b'ready', process_id.encode('utf-8')])
            # ...     
        # ... END LOOP ... !
    except zmq.ZMQError as e: 
        logger.error(f'worker {process_id} => [Exception]zmqerror', e)
    except KeyboardInterrupt as e:
        logger.error(f'worker {process_id} => [Exception]keyboard', e)
    except Exception as e: 
        logger.error(f'worker {process_id} => [Exception]global', e)
    finally: 
        subscriber_socket.close()
        req_socket.close()
        ctx.term()
        logger.info(f'worker {process_id} is closed ...!')
        workers_states_queue.put((process_id, 'dead'))
        
def router_process(router_port, publisher_port, source_images, target, router_readyness, workers_states_queue):
    try:
        ctx = zmq.Context()
        logger.info('the context was created ...')

        router_socket = ctx.socket(zmq.ROUTER)
        publisher_socket = ctx.socket(zmq.PUB)

        logger.info('router and publisher socket was initialized')
       
        router_socket.bind(f'tcp://*:{router_port}')
        publisher_socket.bind(f'tcp://*:{publisher_port}')

        logger.info('router and publisher socket was bound')

        router_socket_poller = zmq.Poller()
        router_socket_poller.register(router_socket, zmq.POLLIN)

        workers_states_map = {}

        print(source_images)
        nb_source_images = len(source_images) 
        counter = 0 
        keep_routing = True 
        router_readyness.set()
        logger.info('router process is ready')
        while keep_routing:
            keep_routing = counter < nb_source_images
            router_events = dict(router_socket_poller.poll(100))
            while not workers_states_queue.empty():
                pid, sts = workers_states_queue.get()
                workers_states_map[pid] = sts 
            if router_socket in router_events.keys():
                if router_events[router_socket] == zmq.POLLIN: 
                    inc = router_socket.recv_multipart()
                    rid, sep, msg, pid = inc 
                    if msg == b'ready':
                        if counter < nb_source_images:                                   
                            contents = json.dumps({
                                'source': source_images[counter],
                                'target': target
                            }).encode() 
                            logger.info(f'router gor {msg} for worker {pid}')
                            router_socket.send_multipart([  # send this request to the first ready worker 
                                rid, # tis id will be used by the router 
                                b'', # this one will deleted by the req socket 
                                b'', # just for convinience
                                contents 
                            ])
                            counter = counter + 1              
        # ...  
        publisher_socket.send_multipart([b'KILL', b''])  # kill all workers
        logger.info('router process pending for worker to finish cleaning ...!')
        start = time.time()
        while any([sts != 'dead' for sts in workers_states_map.values()]):
            current_time = time.time()
            if current_time - start >= 1:
                start = current_time 
                logger.info(workers_states_map)
            if not workers_states_queue.empty():
                pid, sts = workers_states_queue.get()
                workers_states_map[pid] = sts 

            time.sleep(0.001)
        
        logger.info('all worker are free => router can terminate !')

    except zmq.ZMQError as e: 
        logger.error('router => [Exception]zmqerror', e)
    except KeyboardInterrupt as e:
        logger.error('router => [Exception]keyboard', e)
    except Exception as e: 
        logger.error('router => [Exception]global', e)
    finally: 
        publisher_socket.close()
        router_socket.close()
        ctx.term()
        logger.info('router process is closed ...!')

if __name__ == '__main__':
    print(' ... ... ')

    parser = get_parser()
    
    parser.add_argument('--router_port', help='port of the router socket', required=True, type=int)
    parser.add_argument('--publisher_port', help='port of the publisher socket', required=True, type=int)
    parser.add_argument('--nb_workers', help='number of worker', default=4, type=int)
    parser.add_argument('--source', help='path to source images', required=True)
    parser.add_argument('--target', help='path to target', required=True)

    parser_map = to_map(parser)
    
    router_port = parser_map['router_port']
    publisher_port = parser_map['publisher_port']
    source, target = parser_map['source'], parser_map['target']

    router_readyness = mp.Event()
    workers_states_queue = mp.Queue()

    current_dir = get_location(__file__)
    path_to_source = path.join(current_dir, '..', source)
    path_to_target = path.join(current_dir, '..', target)

    source_images = pull_files(path_to_source)
    
    try:
        workers = []
        for idx in range(parser_map['nb_workers']):
            workers.append(
                mp.Process(
                    target=finder_process,
                    args=[router_port, publisher_port, '%03d' % idx, router_readyness, workers_states_queue]
                )
            )
            workers[-1].start()
        
        server = mp.Process(
            target=router_process, 
            args=[router_port, publisher_port, source_images, target, router_readyness, workers_states_queue]
        )
        server.start()

        server.join()
        for wrk in workers:
            wrk.join()

    except KeyboardInterrupt as e:
        logger.error('main => [Exception]keyboard', e)



