from rest_framework import serializers
from ml.models import array,svm
from rest_framework.viewsets import ModelViewSet
from django.contrib.auth.models import User

class arraySerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(read_only=True, label="编号")
    class Meta:
        model = array # 参考模型类生成字段
        fields = "__all__"  # 生成所有字段

    # def validate_body(self, value):
    #     print("value = {}".format(value))
    #
    #
    #     for i in value:
    #         flag = True
    #         if i not in "ATCGUatcgu":
    #             flag = False
    #             break
    #
    #     if flag == False:
    #         raise serializers.ValidationError("The input sequence should all be composed of ATCGU (atcgu), please re-enter")
    #
    #
    #     return value

class svmSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(read_only=True, label="编号")

    class Meta:
        model = svm  # 参考模型类生成字段
        fields = "__all__"

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model=User
        fields=['username','password','email']

    def create(self, validated_data):
        user=User()