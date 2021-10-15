from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, f1_score, accuracy_score


def linearSVC(X, y, testSet_X, testSet_y):
    clf = Pipeline([("scaler", StandardScaler()),
                    ("linear_svc",
                     LinearSVC(C=1, loss="hinge", random_state=42))])
    # 一维数组使用时形式需要进行ravel()转换
    clf.fit(X, y)
    y_pre = clf.predict(testSet_X)
    s = clf.score(testSet_X, testSet_y)
    print("准确率得分：", s)
    # 生成性能报告
    print(classification_report(testSet_y, y_pre))
    # 输出参数
    print(clf.get_params())


def SVC(X, y, testSet_X, testSet_y):
    clf = Pipeline([("scaler", StandardScaler()),
                    ("svc",
                     svm.SVC(kernel='linear'))])
    clf.fit(X, y)
    y_pre = clf.predict(testSet_X)

    s = clf.score(testSet_X, testSet_y)
    # accuracy = accuracy_score(testSet_y, y_pre)
    f1 = f1_score(testSet_y, y_pre)
    # print("准确率得分：",s)
    # print("f1：",f1)

    # 生成性能报告
    report = classification_report(testSet_y, y_pre)
    # print(classification_report(testSet_y, y_pre))

    # 输出参数
    params = clf.get_params()
    w = clf["svc"].coef_
    b = clf["svc"].intercept_
    # print(clf.get_params())
    # print(clf["svc"].coef_)
    # print(clf["svc"].intercept_)

    return s, f1, report, params, w, b