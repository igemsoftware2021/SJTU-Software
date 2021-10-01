from django.db import models
from django.contrib.auth.models import User

# 用于构建二级结构预测接口的模型
class array(models.Model):

    #定义字段是否可以为空,blank 用于字段的HTML表单验证，即判断用户是否可以不输入数据
    body = models.CharField(max_length=100, default= "ATCGATCG",verbose_name='输入序列')
    time = models.DateTimeField(auto_now_add=True,verbose_name="日期")
    result = models.TextField()


    class Meta:
        db_table = 'db_array'  #指定数据库表名
        verbose_name = 'array'  #在admin站点中显示的名称
        verbose_name_plural = verbose_name  #显示的复数名称

# 用于连接SVM二分类部分的模型
class svm(models.Model):
    name = models.CharField(max_length=20,verbose_name="名称")
    body = models.FileField(verbose_name="上传文件",upload_to='file_page')
    time = models.DateTimeField(auto_now_add=True, verbose_name="日期")

    class Meta:
        db_table = 'db_svm'  #指定数据库表名
        verbose_name = 'svm'  #在admin站点中显示的名称
        verbose_name_plural = verbose_name  #显示的复数名称

    def __str__(self):
        return self.name


# Create your models here.
