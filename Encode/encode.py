import numpy as np
import cv2
import math

quantification_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]
     ])

# 保存原图数据的变量
picture = np.empty([512, 512], dtype=np.uint8)
# 保存处理后的图片的变量
processed_picture = np.empty([512, 512], dtype=np.uint8)
# 保存量化后的图像
quantified_blocks = np.empty([4096, 8, 8], dtype=np.float)


# 处理图像
def process(image):
    processing_image = np.float64(image)
    length = processing_image.shape[0]


# Z 形扫描
def zigzag_scan(block):
    return np.concatenate([np.diagonal(block[::-1, :], k)[::(2 * (k % 2) - 1)] for k \
                           in range(1 - block.shape[0], block.shape[0])])


# 将 Z 形扫描结果还原
def reverse_zigzag_scan(chain, num):
    original_block = np.zeros([8, 8], dtype=np.float)
    if num == 2:
        original_block[0][0] = chain[0]
        original_block[0][1] = chain[1]
    if num == 3:
        original_block[0][0] = chain[0]
        original_block[0][1] = chain[1]
        original_block[1][0] = chain[2]
    if num == 5:
        original_block[0][0] = chain[0]
        original_block[0][1] = chain[1]
        original_block[1][0] = chain[2]
        original_block[2][0] = chain[3]
        original_block[1][1] = chain[4]
    if num == 8:
        original_block[0][0] = chain[0]
        original_block[0][1] = chain[1]
        original_block[1][0] = chain[2]
        original_block[2][0] = chain[3]
        original_block[1][1] = chain[4]
        original_block[0][2] = chain[5]
        original_block[0][3] = chain[6]
        original_block[1][2] = chain[7]
    return original_block


def encode(filename: str, k, quantification_on):
    """
    编码端
    :param filename: 要编码的文件名
    :param k: 量化系数
    :param quantification_on: 是否量化（若为 False，则参数 k 无意义）
    :return: Z 形扫描结果
    """
    # 保存原始图片信息
    global picture, quantified_blocks
    # 读取图片
    lena = np.fromfile(filename, dtype=np.uint8)
    # print(lena)
    picture = np.reshape(lena, (512, 512))
    print(picture)
    # 记录图片边长
    length = int(np.sqrt(lena.shape[0]))
    image = np.empty([length, length], dtype=np.uint8)
    for i in range(length):
        for j in range(length):
            image[i][j] = lena[i * length + j]
    # 创建以 8 * 8 块存储图片信息的结构
    blocks = np.empty([int(length * length / 64), 8, 8], dtype=np.float)
    split_image = np.split(image, int(length / 8), 0)
    # 存储图像至 block 结构
    for i in range(0, len(split_image)):
        split_split_image = np.split(split_image[i], int(length / 8), 1)
        for j in range(0, len(split_split_image)):
            blocks[i * int(length / 8) + j] = split_split_image[j]
    # - 128
    for i in range(blocks.shape[2]):
        blocks[:, :, i] = blocks[:, :, i] - 128
    # DCT 变换
    for i in range(blocks.shape[0]):
        blocks[i] = cv2.dct(blocks[i])
        # 量化
        if quantification_on:
            blocks[i] = blocks[i] / (k * quantification_table)
    blocks = np.around(blocks, decimals=0)
    # 将预处理后的 blocks 保存
    quantified_blocks = blocks
    # 用与保存数据序列
    chains = np.empty([int(length * length / 64), 64], dtype=np.float)
    # 将 Z 形扫描后的顺序填充进去
    for i in range(blocks.shape[0]):
        chains[i] = zigzag_scan(blocks[i])
    return chains


def decode(chains, num, keep_whole_blocks, k, quantification_on):
    """
    解码端
    :param chains: 编码端输入的 Z 形扫描结果
    :param num: 该 Z 形扫描保留的位数
    :param keep_whole_blocks: 是否保留 Z 形扫描所有位数（若为 True，则参数 num 无意义）
    :param k: 量化系数
    :param quantification_on: 是否量化（若为 False，则参数 k 无意义）
    :return: 解码图片结果
    """
    global quantified_blocks
    # Debug
    blocks = np.empty([chains.shape[0], 8, 8], dtype=np.float)
    for i in range(0, chains.shape[0]):
        blocks[i] = reverse_zigzag_scan(chains[i], num)
    # 如果是保留所有信息
    if keep_whole_blocks:
        blocks = quantified_blocks
    # 还原 DCT 变换
    for i in range(blocks.shape[0]):
        # 反量化
        if quantification_on:
            blocks[i] = blocks[i] * quantification_table * k
        blocks[i] = cv2.idct(blocks[i])
    # + 128
    for i in range(blocks.shape[2]):
        blocks[:, :, i] = blocks[:, :, i] + 128
    original_pic = np.empty([512, 512], dtype=np.uint8)
    # original_pic 充当原图图片矩阵
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            for k in range(blocks.shape[2]):
                original_pic[int(i / 64) * 8 + j][int(i % 64) * 8 + k] = int(blocks[i][j][k])
    print(original_pic)
    cv2.imshow('src', original_pic)
    cv2.waitKey()
    return original_pic


def calculate_psnr(original_pic, processed_pic):
    """
    计算 PSNR
    :param original_pic: 原图
    :param processed_pic: 解码后的图片
    :return: PSNR 值
    """
    tmp = (original_pic / 255.0 - processed_pic / 255.0) ** 2
    mean_pic = np.mean(tmp)
    return 20 * math.log10(1 / math.sqrt(mean_pic))


if __name__ == '__main__':
    chain = encode('lena.raw', 14, True)
    original_pic = decode(chain, 8, True, 14, True)
    print(calculate_psnr(picture, original_pic))
