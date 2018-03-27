#文件说明

1. libLogoRecog.h 头文件 libLogoRecog.cpp 算法库文件
2. trainWithLogos.cpp 训练程序, 在PC上使用
3. runtimeExample.cpp 手机上加载训练结果和识别的用法示例(同时也是测试数据集效果的代码)
4. matchTest2.py 算法Python原型
5. buildTestList.py buildTrainList.py tlist.txt vlist.txt okr\_nt\_240.txt 测试用的辅助脚本和图片列表等

#训练方法

使用trainWithOneLogo函数, 用字符串输入要识别的模板图片的路径, 和一些准备用来识别的(难度适中,不要太难)的图片, 同样用字符串传入路径, 第三个参数指定训练结果的存储路径

这个过程主要是为了提取logo的特征和识别它需要的最少特征数, 预先存储下来, 实际识别时用哪个logo就加载哪个logo的训练结果(用loadLogos 成员函数)

inverse选项默认开启, 是因为有些logo会有不同底色的变体(比如雀巢和九阳), 要正反提取两种特征才能识别全部情况, 但是这样会让识别时间变长, 如果确定一个logo的形式比较固定,底色不会改变, 可以设为false

具体参考trainWithLogos.cpp 和 tlist.txt buildTrainList.py

目前训练结果的存储和加载都是用文本的形式, 如果在手机上性能不够, 可以自行改成用二进制形式

#识别方法
建立类的实例后, 先执行init()

这里面会有一些参数, 大部分不要动, 如果想加速可以修改des_sz为小于240的数, 比如180,160, 但会明显损失效果. 所有的参数如果不给或者给小于0的数, 则会用头文件中的常量初始化为默认值, 将MAT\_FT 设置为200-400也可以加速, 但是会损失效果.

另外, 同时识别的logo数越少, 速度越快.

前面的blur_det是是否开启模糊识别, 如果开启, 则模糊时不进行识别直接返回, 可能实际使用时总体速度比较快, 但识别成功率会下降, 所以暂时不建议开启(设为false)

然后加载logo的训练结果(字符串传入训练结果文件的路径)

然后用recognize成员函数, 输入是opencv mat形式存储的图像, 最好是短边720的图, 长宽比为960:720,横图竖图均可, 如果不是这个长宽比需要修改代码; 图片为bgr格式(opencv惯例)

返回一个int, 即识别到的logo编号(按loadLogos时的下标, 0开始), 如果是-1, 则是未识别到, 如果是其他负数, 则是有错误(如果开启了模糊识别, -2是模糊)

具体参考runtimeExample.cpp和vlist.txt buildTestList.py
