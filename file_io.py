# coding = utf-8

"""
关于文件处理的相关工具类
"""

import os
import platform


def get_absolute_project_dir():
    """
    获取绝对的项目路径
    :return:
    """
    path = os.getcwd()
    if platform.system() == "Windows":
        path = path[: path.index("MyAll")+len("MyAll")]
    else:
        path = path[: path.index("my-all") + len("my-all")]
    return path



if __name__ == '__main__':
    get_absolute_project_dir()