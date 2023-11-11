'''
CIEDE2000颜色差异度量是一个更加准确的颜色差异指标,它考虑了人眼的视觉感知特性,能够更好地反映实际的颜色差异情况。
具体来说,它将三个颜色属性分别转化到LCH(明度、色度、色调)颜色空间,然后计算两个LCH颜色向量之间的距离。
与欧几里得距离不同,CIEDE2000距离具有对颜色空间的非线性适应性,更能反映人眼在不同颜色和明度下的感知特性。
在Python中,可以使用colormath.color_diff.delta_e_cie2000函数计算CIEDE2000距离。
取色方法：
随机初始化大量的RGB颜色,从中选取N个相互之间的CIEDE2000颜色差异最大的颜色。
'''
import itertools
import random
import colorsys

from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from tqdm import tqdm


def generate_random_colors(num_colors=100):
    """生成指定数量的随机RGB颜色"""
    return [generate_random_color() for _ in range(num_colors)]


def generate_random_color():
    """生成一个随机的RGB颜色"""
    hue = random.uniform(0, 360)
    saturation = random.uniform(0.3, 1)
    value = random.uniform(0.3, 1)
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
    return tuple(int(c * 255) for c in (r, g, b))


def find_max_distinct_colors(colors, num_colors=10):
    # 转换为Lab颜色空间
    lab_colors = [
        LabColor(*rgb_color, illuminant='D50') for rgb_color in colors
    ]

    # 找到距离最大的两个颜色
    max_distance = 0
    max_indexes = None
    for indexes in tqdm(itertools.combinations(range(num_colors), 2)):
        distance = delta_e_cie2000(lab_colors[indexes[0]],
                                   lab_colors[indexes[1]])
        if distance > max_distance:
            max_distance = distance
            max_indexes = indexes

    selected_indexes = set(max_indexes)

    # 选出距离最大的N个颜色
    while len(selected_indexes) < num_colors:
        print(len(selected_indexes))
        distances = []
        for i in range(len(colors)):
            if i not in selected_indexes:
                min_distance = float('inf')
                for j in selected_indexes:
                    distance = delta_e_cie2000(lab_colors[i], lab_colors[j])
                    min_distance = min(min_distance, distance)
                distances.append((min_distance, i))

        distances.sort(reverse=True)
        selected_indexes.add(distances[0][1])

    return [colors[i] for i in selected_indexes]


if __name__ == '__main__':
    generate_num_colors = 2000
    final_num_colors = 150

    colors = generate_random_colors(generate_num_colors)
    final_colors = find_max_distinct_colors(colors,
                                            num_colors=final_num_colors)
    print(final_colors)
