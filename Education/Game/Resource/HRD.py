import simpleguitk as gui
import math
import ssl
import requests
from io import BytesIO
from PIL import Image
ssl._create_default_https_context = ssl._create_unverified_context
# 全局变量
canvas_height = 500           # 画布高度，单位为像素
canvas_width = 400            # 画布宽度，单位为像素
game_over = False             # 游戏是否结束
figure_moving = False         # 是否有运动的人物
figures = {}                  # 所有人物
steps = 0                     # 移动步数
current_figure = None         # 鼠标点中的人物
current_center = []           # 鼠标点中人物的中心坐标
original_point = []           # 鼠标点击的初始位置坐标，用来计算鼠标拖动的方向
speed = 5                     # 人物移动的速度

machao_image = gui.load_image('https://tva1.sinaimg.cn/large/008i3skNgy1gv751qutxsj602s05kglg02.jpg')
zhangfei_image = gui.load_image('https://tva1.sinaimg.cn/large/008i3skNgy1gv751xw643j602s05kwec02.jpg')
zhaoyun_image = gui.load_image('https://tva1.sinaimg.cn/large/008i3skNgy1gv7522wfuwj602s05kmx102.jpg')
huangzhong_image = gui.load_image('https://tva1.sinaimg.cn/large/008i3skNgy1gv7527s66aj602s05kwec02.jpg')
guanyu_image = gui.load_image('https://tva1.sinaimg.cn/large/008i3skNgy1gv752f4yc3j605k02s74702.jpg')
caocao_image = gui.load_image('https://tva1.sinaimg.cn/large/008i3skNgy1gv76o945mxj605k05k74802.jpg')
soldier_image = gui.load_image('https://tva1.sinaimg.cn/large/008i3skNgy1gv752ql3uij602s02s0sj02.jpg')

picture_list = [machao_image,zhangfei_image,zhaoyun_image,huangzhong_image,guanyu_image,caocao_image,soldier_image,soldier_image,soldier_image,soldier_image]
figure_src_size = [[100,200],[100,200],[100,200],[100,200],[200,100],[200,200],[100,100],[100,100],[100,100],[100,100]]
figure_src_center =[[50,100], [50,100], [50,100], [50,100], [100,50],[100,100], [50,50], [50,50], [50,50], [50,50]]
figure_name =["virus1","cell1","cell2","virus","lab","Nano-DNA","dna","dna","dna","dna"]
figure_des_center = [[350,300],[50,300],[350,100],[50,100],[200,250],[200,100],[150,350],[250,350],[50,450],[350,450]]
figure_des_size =[[100,200],[100,200],[100,200],[100,200],[200,100],[200,200],[100,100],[100,100],[100,100],[100,100]]
# Figure类（棋子类）
class Figure:
    def __init__(self, image, src_center, src_size, des_center, des_size, name, move_direction = None):
        self.image = image                     # 棋子图像
        self.src_center = src_center           # 源图像中心坐标
        self.src_size = src_size               # 源图像大小
        self.des_center = des_center           # 画布显示图像中心坐标
        self.des_size = des_size               # 画布显示图像大小
        self.name = name                       # 棋子名称，如“曹操”
        self.move_direction = move_direction   # 移动方向
    def get_des_center(self):
        return self.des_center
    def get_des_size(self):
        return self.des_size
    def get_name(self):
        return self.name
    def set_move_direction(self, direction):
        self.move_direction = direction
    def draw(self, canvas):
        canvas.draw_image(self.image,self.src_center, self.src_size, self.des_center, self.des_size)
    def update(self):
        if self.move_direction =='left':
            self.des_center[0] -= speed
        elif self.move_direction =='right':
            self.des_center[0] += speed
        elif self.move_direction =='up':
            self.des_center[1] -= speed
        elif self.move_direction =='down':
            self.des_center[1] += speed
        else:
            self.des_center[0] = self.des_center[0]
            self.des_center[1] = self.des_center[1]
    def collide(self, other):
        global figure_moving, steps
        h_distance = self.des_center[0]-other.get_des_center()[0]
        h_length = self.des_size[0]/2+other.get_des_size()[0]/2
        v_distance = self.des_center[1]-other.get_des_center()[1]
        v_length = self.des_size[1]/2+other.get_des_size()[1]/2
        if self.move_direction =='left' and h_distance>0 and h_distance <=h_length and math.fabs(v_distance) < v_length :
            #print(self.des_size[0]/2+other.get_des_size()[0]/2)
            figure_moving = True
            self.move_direction = None
            # print(h_distance<0)
            # print(h_distance <=h_length)
            # print(math.fabs(v_distance < v_length))
        elif self.move_direction =='right'and h_distance<0 and -h_distance <=h_length and math.fabs(v_distance) < v_length:
            figure_moving = False
            self.move_direction = None
        elif self.move_direction =='down'and  v_distance<0 and -v_distance <=v_length and math.fabs(h_distance) < h_length :
            figure_moving = False
            self.move_direction = None
            # print(v_distance<0)
            # print(-v_distance <=v_length)
            # print(math.fabs(h_distance < h_length))
        elif self.move_direction =='up'and v_distance>0 and v_distance <=v_length and math.fabs(h_distance) < h_length:
            figure_moving = False
            self.move_direction = None
        else:
            figure_moving = figure_moving
            self.move_direction =self.move_direction
        if self.move_direction =='left' and self.des_center[0]<=self.des_size[0]/2:
            figure_moving = False
            self.move_direction = None
        elif self.move_direction =='right' and self.des_center[0]>=canvas_width-self.des_size[0]/2-1:
            figure_moving = False
            self.move_direction = None
        elif self.move_direction =='up' and self.des_center[1]<=self.des_size[1]/2:
            figure_moving = False
            self.move_direction = None
        elif self.move_direction =='down' and self.des_center[1]>=canvas_height-self.des_size[1]/2-1:
            figure_moving = False
            self.move_direction = None
        else:
            figure_moving = figure_moving
            self.move_direction =self.move_direction
        label_text = "Move number = " + str(steps) + " step"
        label.set_text(label_text)
# # 检查移动与其它静止棋子及边界的碰撞
def check_collide():
    global game_over
    if figure_list[current_figure].get_name()=="Nano-DNA" and figure_list[current_figure].get_des_center() == [200,400]:
        game_over = True
        figure_list[current_figure].set_move_direction("down")
    for p in figure_list:
        if figure_list.index(p) != current_figure :
            figure_list[current_figure].collide(p)
# 绘制全部棋子
def draw_figures(figures, canvas):
    for p in figures:
        p.draw(canvas)

# 绘制游戏结束信息
def draw_game_over_msg(canvas, msg):
    canvas.draw_text(msg, (150, 200), 48, 'Red')
# 鼠标点击事件的处理函数
def mouse_click(pos):
    global current_center,current_figure,original_point,steps
    flag_pos = list(pos)
    for p in figure_list:
        list1 = p.get_des_center()
        list2 =p.get_des_size()
        if flag_pos[0]<=list1[0]+list2[0]/2 and flag_pos[0]>=list1[0]-list2[0]/2:
            if flag_pos[1]<=list1[1]+list2[1]/2 and flag_pos[1]>=list1[1]-list2[1]/2:
                m=figure_list.index(p)
                current_figure = m
                current_center = figure_list[m].get_des_center
                original_point = list(pos)
    steps = steps+1
# 鼠标拖动事件的处理函数
def mouse_drag(pos):
    global figure_moving,steps
    fina_pos = list(pos)
    if current_figure != None:
        figure_moving = True
        if math.fabs(original_point[0]-fina_pos[0])> math.fabs(original_point[1]-fina_pos[1]) and original_point[0]-fina_pos[0]>0:
            figure_list[current_figure].set_move_direction('left')
        elif math.fabs(original_point[0]-fina_pos[0])> math.fabs(original_point[1]-fina_pos[1]) and original_point[0]-fina_pos[0]<0:
            figure_list[current_figure].set_move_direction('right')
        elif math.fabs(original_point[0]-fina_pos[0])<= math.fabs(original_point[1]-fina_pos[1]) and original_point[1]-fina_pos[1]>0:
            figure_list[current_figure].set_move_direction('up')
        else:
            figure_list[current_figure].set_move_direction('down')
# 屏幕刷新事件处理函数
def draw(canvas):
    if game_over == True:
        draw_game_over_msg(canvas, "Success!")
    else:
        draw_figures(figure_list,canvas)
        if current_figure != None:
            check_collide()
            figure_list[current_figure].update()
# 为游戏开始或重新开始初始化全局变量，也是鼠标点击按钮的事件处理函数
# 创建对象
figure_list = []
def start_game():
    global figure_list,picture_list,figure_src_size,figure_src_center,figure_name,figure_des_center,figure_des_size,game_over,figure_moving,steps,current_figure,current_center,original_point
    picture_list = [machao_image,zhangfei_image,zhaoyun_image,huangzhong_image,guanyu_image,caocao_image,soldier_image,soldier_image,soldier_image,soldier_image]
    figure_src_size = [[100,200],[100,200],[100,200],[100,200],[200,100],[200,200],[100,100],[100,100],[100,100],[100,100]]
    figure_src_center =[[50,100], [50,100], [50,100], [50,100], [100,50],[100,100], [50,50], [50,50], [50,50], [50,50]]
    figure_name =["virus1","cell1","cell2","virus","lab","Nano-DNA","dna","dna","dna","dna"]
    figure_des_center = [[350,300],[50,300],[350,100],[50,100],[200,250],[200,100],[150,350],[250,350],[50,450],[350,450]]
    figure_des_size =[[100,200],[100,200],[100,200],[100,200],[200,100],[200,200],[100,100],[100,100],[100,100],[100,100]]
    figure_list = []
    for i in range(10):
        p =Figure(picture_list[i],figure_src_center[i],figure_src_size[i],figure_des_center[i],figure_des_size[i],figure_name[i],move_direction=None)
        figure_list.append(p)
    game_over = False             # 游戏是否结束
    figure_moving = False         # 是否有运动的人物
    steps = 0                     # 移动步数
    current_figure = None         # 鼠标点中的人物
    current_center = []           # 鼠标点中人物的中心坐标
    original_point = []
    label.set_text("Round number = 0")

# 创建窗口初始化画布
frame = gui.create_frame("Bio-game", canvas_width, canvas_height)
label = frame.add_label("Move number = 0 step")
# 注册事件处理函数
frame.set_draw_handler(draw)  # 显示处理，每秒调用draw函数60次
button = frame.add_button('Restart', start_game, 50)  # 鼠标每次点击“重新开始游戏”按钮，调用start_game函数1次
frame.set_mouseclick_handler(mouse_click)  #
frame.set_mousedrag_handler(mouse_drag)
# 启动游戏
start_game()  # 为游戏开始或重新开始初始化全局变量
frame.start()  # 显示窗口

