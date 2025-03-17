# -*- coding: utf-8 -*-
# V2024
from pkgutil import iter_modules
import platform
print ('Python version = ',platform.python_version())
pkname_list = [name for loader, name, ispkg in iter_modules()]

for x in ['numpy', 'scipy', 'matplotlib', 'IPython']:
    if x in pkname_list:
        if hasattr(__import__(x),"__version__"):
            print('%s version = %s' % (x, __import__(x).__version__,))
        else:
            print('%s version = unknown'%(x))
    else:
        print('%s does not exist' % (x,))

# 范雨，20240219
# 大概输出这样的结果就说明配置正确了：
# 版本号略有不同问题不大
# Python version =  3.8.7
# numpy version = 1.19.4
# scipy version = 1.5.4
# matplotlib version = 3.3.2
# IPython version = 7.19.0