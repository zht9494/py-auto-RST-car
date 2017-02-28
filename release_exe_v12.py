#coding=utf-8
#添加对中文注释的支持
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from collections import Counter
import math
import operator
import serial
print 'version 1.2'
#串口初始化
ser = serial.Serial('/dev/tty.usbmodem1421',115200)#引用串口库
ser.write("9")#向下位机发送初始化指令
#初始化结束
nowTime = time.clock()
C = np.pi / 3
last_in_num = 0
near = 17
gar = 2
LOCK = 0
LOCK_INGAR = 0
LOCK_GAR = [[0, 0], [0, 0]]
debug = 0
parall = 0
maxnum = 500
p0x1 = 110
p0x2 = 190
p0y1 = 300
p0y2 = 300
r = 200
cfce = 130
stop = 0
# initial camera & serial
if not debug:
    cap = cv2.VideoCapture(0)
##ser = serial.Serial('/dev/ttyACM0', 19200,timeout = 0)
##ser.write("a")  # unlock motor switch
# multi-processing:1/2
font=cv2.FONT_HERSHEY_SIMPLEX
url = '/Users/zhaohangtian/Pictures/garage99.jpg'
dis_fil = 35
def handler(array):
    #Bubble method冒泡法排序
    for i in range(len(array)):
        max_index = i
        for j in range(i,len(array)):
            if array[max_index][2] < array[j][2]:
                max_index = j

        tmp = array[i]
        array[i] = array[max_index]
        array[max_index] = tmp

def angle_bisector(array1,array2):
    #求角平分线
    print "array1"
    print array1
    print "array2"
    print array2
    x1 = array1[0][0]
    y1 = array1[0][1]
    x2 = array1[1][0]
    y2 = array1[1][1]
    x3 = array2[0][0]
    y3 = array2[0][1]
    x4 = array2[1][0]
    y4 = array2[1][1]
    #print x1,y1,x2,y2,x3,y3,x4,y4
    k1 = (y1 - y2) / (x1 - float(x2))
    b1 = y1 - k1 * x1
    k2 = (y3 - y4) / (x3 - float(x4))
    b2 = y3 - k2 * x3
    #print k1,b1,k2,b2
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    k31 = (y1 - y4) / (x1 - float(x4))
    k32 = (y2 - y3) / (x2 - float(x3))
    k3 = (k31 + k32)/2
    b3 = y - k3 * x
    print "x=%f, y=%f, k=%f, b=%f" % (x,y,k3,b3)
    return [x,y,k3,b3]

def rebld_gar(array,k,b):
    print k,b
    el = 3
    arrayList_up = [([0] * 3) for p in range(gar+1)]
    arrayList_dn = [([0] * 3) for q in range(gar+1)]
    garage = [0] * gar
    center = [0] * gar
    p, q = 0, 0
    for i in range(0, len(array)):
        print array[i][1], (k * array[i][0] + b)
        if array[i][1] > (k * array[i][0] + b):
            arrayList_up[p] = array[i]
            p += 1
            print "p = %d" %p
        else:
            arrayList_dn[q] = array[i]
            q += 1
        print "q = %d" % q
        if abs(q-p) > 1:
            return 0,0,0,0
    #return arrayList_up,arrayList_dn
    print "garage"
    for a in range(0,gar):
        garage[a] = np.array([[[arrayList_up[a][0]-el, arrayList_up[a][1]-el], [arrayList_up[a+1][0]+el, arrayList_up[a+1][1]-el], [arrayList_dn[a+1][0]+el, arrayList_dn[a+1][1]+el], [arrayList_dn[a][0]-el, arrayList_dn[a][1]+el]]], dtype=np.int32)
        center[a] = [(arrayList_up[a][0]+arrayList_up[a+1][0]+arrayList_dn[a][0]+arrayList_dn[a+1][0])/4 , (arrayList_up[a][1]+arrayList_up[a+1][1]+arrayList_dn[a][1]+arrayList_dn[a+1][1])/4]
    print garage
    print "center"
    print center
    return garage,center,arrayList_dn,arrayList_up

def ard_comn(midx):
    print "center-x: %d" % midx
    in_num = 9 + (midx - 150)
    #in_num = 9 + int((midx - 150)/float(1.5))
    print "ref: %d" % in_num
    if in_num >= 16:
        in_num = 16
    if in_num <= 1:
        in_num = 1
    if (in_num != last_in_num) and (not stop):
        print 'angleNum:    %d' % in_num
        if in_num == 1:
            ser.write("1")
        if in_num == 2:
            ser.write("2")
        if in_num == 3:
            ser.write("3")
        if in_num == 4:
            ser.write("4")
        if in_num == 5:
            ser.write("5")
        if in_num == 6:
            ser.write("6")
        if in_num == 6:
            ser.write("7")
        if in_num == 8:
            ser.write("8")
        if in_num == 9:
            ser.write("9")
        if in_num == 10:
            ser.write("0")
        if in_num == 11:
            ser.write("-")
        if in_num == 12:
            ser.write("=")
        if in_num == 13:
            ser.write("[")
        if in_num == 14:
            ser.write("]")
        if in_num == 15:
            ser.write(";")
        if in_num == 16:
            ser.write(",")
        ser.write("gr")

flags = 0
tick = 0
while True: # multi-processing:0/2
    print "---------main---------"
    #if cap.isOpened():
    if 1:
        warp = []
        startTime = time.clock()
        if time.clock() - nowTime > 0.1:
            ser.write("s")
        nowTime = time.clock()
        if debug == 1:
            img = cv2.imread(url)
            warp = cv2.resize(img, (300, 240), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('input', warp)
        else:
            ret, frame = cap.read()
            img = cv2.resize(frame, (300,240), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('input', img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#future maybe have yellow
            rows, cols = gray.shape
            pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
            pts2 = np.float32([[0, 75], [300, 75], [90, 240], [210, 240]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warp = cv2.warpPerspective(img, M, (300, 300))
        '''
        cv2.imshow('output0', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        '''
        # endTime = time.clock()
        # print 'run-0 time: %f' % abs(startTime - endTime)
        # startTime = time.clock()
        # 2-D correct
        '''
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        pts2 = np.float32([[90, 0], [210, 0], [0, 240], [300, 240]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warp = cv2.warpPerspective(img, M, (300, 300))
        '''
        #warp = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('output_warp', warp)
        blur = cv2.GaussianBlur(warp, (5, 5), 0)
        # CANNY (1.Synthetic wave 2.Second order guidance)
        edges = cv2.Canny(blur, 80, 240, apertureSize=3)
        cv2.imshow('output2', edges)
        edges_copy = edges.copy()
        # HOUGHLINES (1.Cosmetic standard 2.Accumulator)
        # minLineLength,MaxLineGap
        if LOCK_INGAR == 0:
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 40, 40)  # 40,40;15
            # print lines
            # continue
            if lines is None:
                print 'lines is None po0'
                continue

            # cv2.imshow('edges',edges)
            # print 'P1 ok'
            # print lines.size
            border_del = []
            # print 'P2 gogo'
            # delete the vision-edge lines
            for i in range(0, (lines.size) / 2):
                rho = round(lines[0][i][0])
                theta = round(lines[0][i][1], 1)
                # print 'rho:%f' % rho
                # print 'theat:%f' % theta
                if 30 <= rho <= 40 or 69 < rho <= 79 or 282 <= rho <= 316 or 234 <= rho <= 244:
                    if 1.5 <= theta <= 1.6 or 2.6 <= theta <= 2.7 or 0.3 <= theta <= 0.6:
                        border_del.append(i)
                        # print 'deleted'
            lines = np.delete(lines, border_del, 1)
            # print lines
            # continue
            D2_coordinate0 = []
            # testTime = time.clock()
            # print 'run-0.5 time: %f' % abs(startTime - testTime)
            if lines.size == 0:
                continue
            print lines.size
            for m in range(0, (lines.size) / 2):
                for n in range(0, m):
                    # Taking into account the efficiency, approximate to the decimal point after one
                    rho1 = round(lines[0][m][0], 1)
                    threta1 = round(lines[0][m][1], 1)
                    rho2 = round(lines[0][n][0], 1)
                    threta2 = round(lines[0][n][1], 1)
                    if abs(threta1 - threta2) > C:
                        # set mininum angle
                        # print('threta1 - threta2 = %f') % (threta1 - threta2)
                        a1 = np.cos(threta1)
                        b1 = np.sin(threta1)
                        a2 = np.cos(threta2)
                        b2 = np.sin(threta2)
                        px = ((rho2 * b1 - rho1 * b2) / (a2 * b1 - a1 * b2))
                        py = ((rho1 - a1 * px) / b1)
                        # print px,py
                        if 5 < px < 295 and 5 <= py <= 295 and (not math.isinf(px)) and (not math.isinf(py)):
                            D2_coordinate0.append((int(px) / 2 * 2, int(py) / 2 * 2))
                            cv2.circle(edges, (int(px), int(py)), 6, (255, 255, 255), 1)
                            # print('x=%d y=%d') % (px, py)
            testTime = time.clock()
            print 'run-1.4 time: %f' % abs(startTime - testTime)
            start = time.clock()
            surplas = []
            # overlap = []
            # continue
            myList = [([0] * 3) for i in range(maxnum)]
            f1 = 0
            for i in range(len(D2_coordinate0)):
                for j in range(i - 1):
                    if abs(D2_coordinate0[i][0] - D2_coordinate0[j][0]) < dis_fil and \
                                    abs(D2_coordinate0[i][1] - D2_coordinate0[j][1]) < dis_fil:
                        key = 0
                        for k in range(len(myList)):
                            if myList[k] == [0, 0, 0]:
                                break
                            if myList[k][0] == D2_coordinate0[j][0] and myList[k][1] == D2_coordinate0[j][1]:
                                myList[k][2] += 1
                                key = 1
                        if key == 0:
                            myList[f1][0] = D2_coordinate0[j][0]
                            myList[f1][1] = D2_coordinate0[j][1]
                            myList[f1][2] = +1
                            f1 += 1
            myList = [ele for ele in myList if ele != [0, 0, 0]]
            # handler(myList)
            myList.sort(key=operator.itemgetter(2), reverse=True)
            print '/myList/'
            print myList
            print '///'
            #for i in range(0, 8):
            #    cv2.circle(edges, (myList[i][0], myList[i][1]), 7, (155, 155, 255), thickness=-1, lineType=8, shift=0)
            # overlap = []
            siftout = []
            for i in range(0, gar * 2 + 2):
                overlap = []
                # handler(myList)
                print i
                if len(myList) == 0:
                    continue
                siftout.append(myList[0])
                # print "now myList"
                # print myList
                for j in range(0, len(myList)):
                    if (abs(myList[0][0] - myList[j][0]) < dis_fil) and (abs(myList[0][1] - myList[j][1]) < dis_fil):
                        overlap.append(j)
                myList = np.delete(myList, overlap, 0)
            print "siftout"
            print siftout
            print '///'
            for i in range(len(siftout)):
                cv2.circle(edges, (siftout[i][0], siftout[i][1]), 7, (100, 155, 255), thickness=-1, lineType=8, shift=0)
                cv2.putText(edges, str(siftout[i]), (siftout[i][0], siftout[i][1]), font, 0.3, (200, 200, 255), 1)
            # print siftout

            if LOCK == 0:
                siftout.sort(key=operator.itemgetter(0), reverse=True)
                # print siftout
                if abs(siftout[0][1] - siftout[len(siftout) - 1][1]) > parall:
                    ed_p1 = [siftout[0], siftout[len(siftout) - 1]]
                    ed_p2 = [siftout[1], siftout[len(siftout) - 2]]
                else:
                    ed_p1 = [siftout[0], siftout[len(siftout) - 2]]  # ed_p1[[x1,y1,p1],[x2,y2,p2]]
                    ed_p2 = [siftout[1], siftout[len(siftout) - 1]]
                cv2.line(edges, (ed_p1[0][0], ed_p1[0][1]), (ed_p1[1][0], ed_p1[1][1]), (255, 255, 255), 4)
                cv2.line(edges, (ed_p2[0][0], ed_p2[0][1]), (ed_p2[1][0], ed_p2[1][1]), (255, 255, 255), 4)
                cv2.imshow('output_temple', edges)
                ag_bc = angle_bisector(ed_p1, ed_p2)  # [x,y,k3,b3]
                if math.isnan(ag_bc[3]) or math.isnan(ag_bc[2]):
                    print "error:04"
                    continue
                cv2.line(edges, (0, int(ag_bc[3])), (300, int(300 * ag_bc[2] + ag_bc[3])), (255, 255, 255), 4)
                rgar, cent, aup, adn = rebld_gar(siftout, ag_bc[2], ag_bc[3])
                if rgar == 0 or cent == 0:
                    print "error:05"
                    continue
                for a in range(0, len(rgar)):
                    cv2.fillPoly(edges_copy, rgar[a], (177, 177, 177))
                    cv2.fillPoly(edges_copy, rgar[a], (177, 177, 177))
                    cv2.putText(edges_copy, ("gar:" + str(a + 1)), (cent[a][0] - 25, cent[a][1]), font, 0.5, (0, 0, 0),
                                1)
                end = time.clock()
                print (end - start)
                cv2.imshow('output3', edges)
                cv2.imshow('output4', edges_copy)

                input_A = int(raw_input("INPUT GARAGE NUMBER : "))
                if (not 0 < input_A <= gar):
                    for j in range(0, 3):
                        print "SORRY , PLEASE INPUT AGAIN :("
                    continue
                for j in range(0, 3):
                    print "LOCKED: GARAGE " + str(input_A) + " :)"
                LOCK = input_A
                for i in range(0, 2):
                    LOCK_GAR[i][0] = aup[input_A - 1 + i][0]
                    LOCK_GAR[i][1] = aup[input_A - 1 + i][1]
            if siftout is None:
                print "siftout IS NULL"
                break
            print siftout
            for j in range(0, 2):
                for i in range(0, len(siftout)):
                    if (abs(siftout[i][0] - LOCK_GAR[j][0]) < 20) and (abs(siftout[i][1] - LOCK_GAR[j][1]) < 20):
                        print "tracking!(0)"
                        LOCK_GAR[j][0] = (siftout[i][0] + LOCK_GAR[j][0]) / 2
                        LOCK_GAR[j][1] = (siftout[i][1] + LOCK_GAR[j][1]) / 2
        #if max(LOCK_GAR[0][1],LOCK_GAR[1][1]) > 160:
        LOCK_INGAR = 1
        if LOCK_INGAR == 1:
            corners = cv2.goodFeaturesToTrack(edges, 20, 0.1, 20)
            tmp_group = []
            for i in corners:
                x, y = i.ravel()
                cv2.circle(edges_copy, (x, y), 5, 255, -2)
                tmp_group.append([x,y])
            for j in range(0, 2):
                for i in range(0, len(tmp_group)):
                    if (abs(tmp_group[i][0] - LOCK_GAR[j][0]) < near) and (abs(tmp_group[i][1] - LOCK_GAR[j][1]) < near):
                        print "tracking(1)"
                        LOCK_GAR[j][0] = int(tmp_group[i][0] + LOCK_GAR[j][0]) / 2
                        LOCK_GAR[j][1] = int(tmp_group[i][1] + LOCK_GAR[j][1]) / 2

        print LOCK_GAR


        cv2.circle(edges_copy, (LOCK_GAR[0][0], LOCK_GAR[0][1]), 6, (100, 155, 255), thickness=-1, lineType=8,
                       shift=0)
        cv2.circle(edges_copy, (LOCK_GAR[1][0], LOCK_GAR[1][1]), 6, (100, 155, 255), thickness=-1, lineType=8,
                       shift=0)
        print "!!!!!!!"
        if max(LOCK_GAR[0][1],LOCK_GAR[1][1]) > 210:
            print "stop!!"
            stop = 1
        temple = 0
        temple1 = 0
        last_in_num = 0


        # print 'run-1.8 time: %f' % abs(startTime - testTime)
        # -> find near -> add & average ---> delte(token +1) -> EXECUTE_AGAIN


        # testTime = time.clock()
        # print 'run-1.5 time: %f' % abs(startTime - testTime)

        compare1 = 300
        compare2 = 300
        position_1 = 0
        position_2 = 0

        p1x = LOCK_GAR[1][0]
        p1y = LOCK_GAR[1][1]
        p2x = LOCK_GAR[0][0]
        p2y = LOCK_GAR[0][1]
        inPoint = [[p1x, p1y], [p2x, p2y]]
        setPoint = [[p0x1, p0y1], [p0x2, p0y2]]
        solution_x = [0, 0]
        solution_y = [0, 0]
        for i in range(0, 2):
            if (inPoint[i][0] - float(setPoint[i][0])) == 0:
                continue
            slope1 = -1 / ((inPoint[i][1] - float(setPoint[i][1])) / (inPoint[i][0] - float(setPoint[i][0])))
            heb1 = (inPoint[i][1] + float(setPoint[i][1])) / 2 - slope1 * (
                (inPoint[i][0] + float(setPoint[i][0])) / 2)
            a = 1 + slope1 ** 2
            b = 2 * slope1 * (heb1 - setPoint[i][1]) - 2 * setPoint[i][0]
            c = setPoint[i][0] ** 2 + (heb1 - setPoint[i][1]) ** 2 - r ** 2
            judge = abs(b ** 2 - 4 * a * c)
            solution_x[i] = (-b + judge ** 0.5) / (2 * a)
            solution_y[i] = slope1 * solution_x[i] + heb1
            # print solution_x,solution_y
            cv2.circle(edges_copy, (int(solution_x[i]), int(solution_y[i])), r, (255, 255, 255), 3)
        #cfce = min(p1y, p2y) - 10
        Midx1 = (-(r ** 2 - (cfce - solution_y[0]) ** 2) ** 0.5 + solution_x[0])
        Midx2 = (-(r ** 2 - (cfce - solution_y[1]) ** 2) ** 0.5 + solution_x[1])
        Midx = (Midx1 + Midx2) / 2
        if math.isnan(Midx) or math.isinf(Midx):
            Midx = temple
        if temple == 0:
            temple = Midx
        if temple1 == 0:
            temple1 = temple
        Midx = (temple * 0.7 + Midx + 0.3 * temple1) / 2
        temple1 = temple

        ard_comn(int(Midx))
        # print 'midx1:%f' % Midx1
        # print 'midx2:%f' % Midx2
        # print 'midx:%f' % Midx
        # queue3.put(1)
        #cv2.circle(edges, (int(Midx1), 75), 5, (255, 255, 255), -3)
        #cv2.circle(edges, (int(Midx2), 75), 5, (255, 255, 255), -3)
        cv2.circle(edges_copy, (int(Midx), cfce), 16, (255, 255, 255), 4)
        # cv2.imwrite('/home/cutoff.jpg', edges)

        cv2.imshow('output3', edges)
        cv2.imshow('output4', edges_copy)
        endTime = time.clock()
        print 'run time: %f' % abs(startTime - endTime)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

