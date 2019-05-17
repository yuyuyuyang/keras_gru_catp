#-*- coding:utf-8 -*-
import tornado.web
import tornado.ioloop
from flask import request
import base64
from english_re.model_test import pred_num
import json
import optparse
import tornado.httpserver
from concurrent.futures import ThreadPoolExecutor
import logging
from logging.handlers import TimedRotatingFileHandler
import os

def getLogger(strPrefixBase):
    strPrefix = "%s%d" % (strPrefixBase, os.getpid())
    logger = logging.getLogger("RELA_BAIKE")
    logger.propagate = False
    handler = TimedRotatingFileHandler(strPrefix, 'H', 1)
    handler.suffix = "%Y%m%d_%H%M%S.log"
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

#定义处理类型
class IndexHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(20)

    #添加一个处理post请求方式的方法
    @tornado.gen.coroutine
    def post(self):
        #向响应中，添加数据
        data = self.request.body
        # print(data)
        # print(data.keys())
        image_bytes = base64.b64encode(data).decode()
        res = pred_num(image_bytes)
        t = {'Data': res}
        return self.write(t)

if __name__ == '__main__':


    # serverRelaBaike = rela_baike_server.getRelaBaikeServer()
    # logger = getLogger(g_log_prefix)
    #创建一个应用对象
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)], autoreload=False, debug=False)

    server_host = "0.0.0.0"
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.bind(17778)
    # print(tornado.ioloop.IOLoop.initialize())
    http_server.start(1)
    # app.listen(server_port, server_host)
    try:
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()

