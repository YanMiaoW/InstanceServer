import torch
from flask import request, Flask, make_response
import json
import numpy as np
import cv2
import base64
from Operation import Operation
from moviepy.editor import *
from flask_cors import CORS
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str

app = Flask(__name__)


def setResponseHeaders(res):
    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Credentials'] = 'true'
    res.headers['Access-Control-Allow-Methods'] = '*'
    res.headers['Access-Control-Allow-Headers'] = '*'
    res.headers['Access-Control-Expose-Headers'] = '*'
    return res

# @app.errorhandler(Error)
# def handle_error(error):
#     response = jsonify(error.to_response())
#     response.status_code = error.response_code
#     return setResponseHeaders(response)


@app.route("/", methods=['OPTIONS'])
def optionRoute():
    print('option')
    res = make_response(' ')
    res.status_code = 200
    return setResponseHeaders(res)


@app.after_request
def af_request(resp):
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp


def func(request):
    f = request.files['image'].stream
    f.seek(0)
    data = f.read()
    data = np.fromstring(data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image

@app.route("/", methods=['POST'])
def get_frame():

    upload_file = request.data
    upload_file = json.loads(upload_file)
    image_data = upload_file['recognize_img']

    if isinstance(image_data, list):
        image_data = image_data[0]
    if isinstance(image_data, dict):
        if 'size' in image_data:
            image_size = image_data['size']
        image_data = image_data['media_data']

        # print(image_data.keys())

    # balance = upload_file['balance_flag']
    # acne = upload_file['acne_flag']
    # acne_res = upload_file['acne_res_flag']
    # full = upload_file['full_flag']
    name = upload_file['name'].lower()
    flag = upload_file['mode_flag']
    print(flag)
    try:
        img_decode_ = image_data.encode('ascii')  # ascii编码
        img_decode = base64.b64decode(img_decode_)  # base64解码
        img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
        img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式
    except:
        np.save('/root/upload_image_data.npy', image_data)
        img_np = np.array(image_data)

        img = cv2.imdecode(img_np.astype(np.uint8), cv2.IMREAD_COLOR)  # 转为OpenCV形式
        # img = img_np.reshape(image_size[0], image_size[1], 3)
        # np.save('/root/test_img_np.npy', img_np)
    global op
    if '.mp4' in name:
        result_path = process_mp4(img_decode, os.path.join('/root', name), op, balance, acne)
        processed_string = getByte(result_path)
        data = {'image': processed_string, 'type': 'mp4'}
        json_data = json.dumps(data)
        return json_data
    else:


        #cv2.imwrite('/root/acne_upload_image.png', img)
        #img = op.process(img, full, balance, acne, acne_res)
        # cv2.imwrite('./input_image.png', img)
        img = op.process(img.astype(np.uint8), flag= flag)
        # cv2.imwrite('./output_image.png', img)
        if img is not None:
            # cv2.imwrite('/root/result.png', img)
            if 'from' in upload_file and upload_file['from'] == 'web':
                processed_string = ndarray2base64(img)
                # np.save('/root/data_size.npy', processed_string)
                # np.save('/root/data_size_string.npy', img.reshape(-1).tostring())
                print('图片处理完成')
            elif 'from' in upload_file and upload_file['from'] == 'client':
                # processed_string = img.reshape(-1).tolist()
                processed_string = img.reshape(-1).tostring()
                print('结果返回客户端')
                return processed_string
            elif 'from' in upload_file and upload_file['from'] == 'client_encode':
                encode_img = cv2.imencode('.png', img)[1]
                processed_string = np.array(encode_img).reshape(-1).tostring()
                print('结果返回客户端(编码压缩)')
                return processed_string
            else:
                processed_string = base64.b64encode(img).decode('utf-8')
            data = {'image': processed_string, 'shape': img.shape}
            json_data = json.dumps(data)

            return json_data
        else:
            return 'failed'

@app.route("/hi", methods=['GET'])
def get_hi():
    res = make_response('hello')
    res.status_code = 200
    print('say hi')
    return setResponseHeaders(res)

def ndarray2base64(img):
    encode_img = cv2.imencode('.png', img)[1]
    str_img = np.array(encode_img).tostring()
    return str(base64.b64encode(str_img), encoding='utf-8')


if __name__ == "__main__":
    global op
    op = Operation()
    app.run("0.0.0.0", port=5006, debug=False)
    CORS(app)
