'''
https://github.com/kong36088/BaiduImageSpider
百度搜索引擎单线程图片爬虫
'''

import os
import re
import urllib
import json
import socket
import urllib.request
import urllib.parse
import urllib.error

import time

from tqdm import tqdm

socket.setdefaulttimeout(5)


class Crawler:

    def __init__(self, t=0.1):
        # t 下载图片时间间隔
        self.time_sleep = t
        self.start_amount = 0
        self.amount = 0
        self.per_page = 0
        self.counter = 0
        self.headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0',
            'Cookie': ''
        }
        self.save_root_path = ''

    # 获取后缀名
    def get_suffix(self, name):
        m = re.search(r'\.[^\.]*$', name)
        if m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    def handle_baidu_cookie(self, original_cookie, cookies):
        """
        :param string original_cookie:
        :param list cookies:
        :return string:
        """
        if not cookies:
            return original_cookie
        result = original_cookie
        for cookie in cookies:
            result += cookie.split(';')[0] + ';'
        result.rstrip(';')
        return result

    # 保存图片
    def save_image(self, rsp_data, word):
        save_image_folder_path = os.path.join(self.save_root_path, word)

        if not os.path.exists(save_image_folder_path):
            os.makedirs(save_image_folder_path)

        # 判断名字是否重复，获取图片长度
        self.counter = len(os.listdir(save_image_folder_path)) + 1
        for image_info in rsp_data['data']:
            try:
                if 'replaceUrl' not in image_info or len(
                        image_info['replaceUrl']) < 1:
                    continue
                obj_url = image_info['replaceUrl'][0]['ObjUrl']
                thumb_url = image_info['thumbURL']
                url = 'https://image.baidu.com/search/down?tn=download&ipn=dwnl&word=download&ie=utf8&fr=result&url=%s&thumburl=%s' % (
                    urllib.parse.quote(obj_url), urllib.parse.quote(thumb_url))
                time.sleep(self.time_sleep)
                suffix = self.get_suffix(obj_url)
                # 指定UA和referrer，减少403
                opener = urllib.request.build_opener()
                opener.addheaders = [
                    ('User-agent',
                     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
                     ),
                ]
                urllib.request.install_opener(opener)
                # 保存图片
                filepath = os.path.join(save_image_folder_path,
                                        str(self.counter) + str(suffix))
                urllib.request.urlretrieve(url, filepath)
                if os.path.getsize(filepath) < 5:
                    print("下载到了空文件，跳过!")
                    os.unlink(filepath)
                    continue
            except urllib.error.HTTPError as urllib_err:
                print(urllib_err)
                continue
            except Exception as err:
                time.sleep(1)
                print(err)
                print("产生未知错误，放弃保存")
                continue
            else:
                print(f"类别:{word}+1图,已有{str(self.counter)}张图")
                self.counter += 1

        return

    # 获取图片
    def get_images(self, word):
        search = urllib.parse.quote(word)
        # pn int 图片数
        pn = self.start_amount
        while pn < self.amount:
            url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%s&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word=%s&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn=%s&rn=%d&gsm=1e&1594447993172=' % (
                search, search, str(pn), self.per_page)
            # 设置header防403
            try:
                time.sleep(self.time_sleep)
                req = urllib.request.Request(url=url, headers=self.headers)
                page = urllib.request.urlopen(req)
                self.headers['Cookie'] = self.handle_baidu_cookie(
                    self.headers['Cookie'],
                    page.info().get_all('Set-Cookie'))
                rsp = page.read()
                page.close()
            except UnicodeDecodeError as e:
                print(e)
                print('-----UnicodeDecodeErrorurl:', url)
            except urllib.error.URLError as e:
                print(e)
                print("-----urlErrorurl:", url)
            except socket.timeout as e:
                print(e)
                print("-----socket timout:", url)
            else:
                # 解析json
                rsp_data = json.loads(rsp, strict=False)
                if 'data' not in rsp_data:
                    print("触发了反爬机制，自动重试！")
                else:
                    self.save_image(rsp_data, word)
                    # 读取下一页
                    print("下载下一页")
                    pn += self.per_page
        print("下载任务结束")

        return

    def start(self,
              word,
              total_page=1,
              start_page=1,
              per_page=30,
              save_root_path=''):
        """
        爬虫入口
        :param word: 抓取的关键词
        :param total_page: 需要抓取数据页数 总抓取图片数量为 页数 x per_page
        :param start_page:起始页码
        :param per_page: 每页数量
        :return:
        """
        self.per_page = per_page
        self.start_amount = (start_page - 1) * self.per_page
        self.amount = total_page * self.per_page + self.start_amount
        self.save_root_path = save_root_path
        self.get_images(word)


if __name__ == '__main__':
    save_root_path = r'/root/autodl-tmp/result'
    word_list = [
        "人",
        "食物",
    ]

    for per_word in tqdm(word_list):
        try:
            save_image_folder_path = os.path.join(save_root_path, per_word)
            if os.path.exists(save_image_folder_path):
                continue

            # 抓取延迟为 0.05
            crawler = Crawler(0.05)
            # "抓取关键词', "需要抓取的总页数", "起始页数", "每页抓取的图片数量"
            crawler.start(per_word, 2, 2, 10, save_root_path)
        except Exception:
            print(f'{per_word} 异常结束!')
            continue
