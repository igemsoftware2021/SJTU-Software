from rest_framework.generics import GenericAPIView
from rest_framework.parsers import JSONParser
from django import http
from ml.models import array, svm
from rest_framework.response import Response
from rest_framework import status
from ml.serializer import arraySerializer, svmSerializer

import time
from model_backend.onebyone import get_result
# from backend.GNN_model_pred import get_pred_GNN_tri,get_pred_GNN_binary

"""
二级结构预测部分
"""
class arraysGenericAPIView(GenericAPIView):
    """
        可以接收JSON内容POST请求的视图。
    """
    parser_classes = (JSONParser,)

    # 公共属性
    queryset = array.objects.all()
    serializer_class = arraySerializer

    def get(self, request):
        # 查询所有
        arrays = self.get_queryset()

        # 对象转换
        serializr = self.get_serializer(instance=arrays, many=True)

        # 返回响应
        return Response(serializr.data)

    def post(self, request):
        # 获取参数
        data_dict = request.data
        body = data_dict["body"]

        for i in body:
            flag = True
            if i not in "ATCGUatcgu":
                flag = False
                break

        if flag == False:
            return Response("The input sequence should all be composed of ATCGU (atcgu), please re-enter",status=status.HTTP_400_BAD_REQUEST)

        outcome = get_result(body)
        data_dict["result"] = str(outcome).replace('\n','')


        # 序列化器
        serializer = self.get_serializer(data=data_dict)

        # 校验,入库
        serializer.is_valid(raise_exception=True)
        serializer.save()

        # 此处再调用模型进行处理
        print(serializer.data["body"], "\n", serializer.data["result"])
        return Response(serializer.data, status=status.HTTP_201_CREATED, )

class arrayGenericAPIView(GenericAPIView):
    """
            可以接收JSON内容POST请求的视图。
        """
    parser_classes = (JSONParser,)

    # 公共属性
    queryset = array.objects.all()
    serializer_class = arraySerializer

    def get(self,request,pk):
        # 获取
        array = self.get_object()  # 根据id到queryset中取出对象

        #序列化
        serializer = self.get_serializer(instance=array)

        #返回
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk):
        #删除
        self.get_object().delete()

        #返回
        return Response(status=status.HTTP_204_NO_CONTENT)

"""
SVM分类部分
"""
class svmsGenericAPIView(GenericAPIView):

    # 公共属性
    serializer_class = svmSerializer
    queryset = svm.objects.all()

    def get(self, request):
        # 查询所有
        svms = self.get_queryset()

        # 对象转换
        serializr = self.get_serializer(instance=svms, many=True)

        # 返回响应
        return Response(serializr.data)


    def post(self, request):
        file = request.data
        file_name=str(file)
        serializer = svmSerializer(data=file)

        serializer.is_valid(raise_exception=True)
        serializer.save()

        parameter = serializer.data["body"]

        # 此处再调用模型进行处理
        # print(serializer.data["body"])

        return Response(serializer.data, status=status.HTTP_201_CREATED, )

class svmGenericAPIView(GenericAPIView):
    """
            可以接收JSON内容POST请求的视图。
        """
    parser_classes = (JSONParser,)

    # 公共属性
    queryset = svm.objects.all()
    serializer_class = svmSerializer

    def get(self,request,pk):
        # 获取
        array = self.get_object()  # 根据id到queryset中取出对象

        #序列化
        serializer = self.get_serializer(instance=array)

        #返回
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk):
        #删除
        self.get_object().delete()

        #返回
        return Response(status=status.HTTP_204_NO_CONTENT)

# Create your views here.
