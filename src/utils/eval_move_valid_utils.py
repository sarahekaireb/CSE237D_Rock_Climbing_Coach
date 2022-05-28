#! /usr/bin/python

import sys
import move_validity_utils
import pandas as pd
import os

def eval_route_color_first(climb_holds_used, holds, colors):
	route_color_pred = move_validity_utils.getColorRoute(climb_holds_used, holds, colors, mode = 'first')
	return route_color_pred

def eval_route_color_freq(climb_holds_used, holds, colors):
	route_color_pred = move_validity_utils.getColorRoute(climb_holds_used, holds, colors, mode = 'freq')
	return route_color_pred

def eval_route_color_hovered(climb_holds_used, holds, colors):
	route_color_pred = move_validity_utils.getColorRoute(climb_holds_used, holds, colors, mode = 'hovered')
	return route_color_pred

def eval_route_color(route_color, VIDEO_PATH, HOLDS_PATH):
	_, climb_holds_used, holds, colors, _, _, _, _, significances = move_validity_utils.process_video(VIDEO_PATH, HOLDS_PATH)
	first_pred = eval_route_color_first(climb_holds_used, holds, colors)
	freq_pred = eval_route_color_freq(climb_holds_used, holds, colors)
	hovered_pred = eval_route_color_hovered(climb_holds_used, holds, colors)
	print("route color: ", route_color)
	print("first_pred: ", first_pred)
	print("freq_pred: ", freq_pred)
	print("hovered_pred: ", hovered_pred)
	return first_pred, freq_pred, hovered_pred
	

def eval_route_color_all(fn, vid_path):
	data = pd.read_csv(fn, sep=" ")
	display(data)
	first_preds = []
	freq_preds = []
	hovered_preds = []
	for i, name in enumerate(data['NAME']):
		video_path = vid_path + name + '/climb.mp4'
		holds_path = vid_path + name + '/holds.jpg'
		first_pred, freq_pred, hovered_pred = eval_route_color(data['COLOR_ROUTE'][i], video_path, holds_path)
		first_preds.append(first_pred)
		freq_preds.append(freq_pred)
		hovered_preds.append(hovered_pred)
	return data, first_preds, freq_preds, hovered_preds

def calcAccuracy(truth, preds):
    correct = 0
    for i in range(len(truth)):
        if truth[i] == preds[i]:
            correct += 1
    return correct / len(truth)

def generateRouteColorTable(data, first_preds, freq_preds, hovered_preds):
	mydf = pd.DataFrame(zip(data['COLOR_ROUTE'], first_preds, freq_preds, hovered_preds))
	mydf.columns = ['Truth', 'First', 'Used', 'Hovered']
	mydf.index = data['NAME']

	a1 = str(round(calcAccuracy(data['COLOR_ROUTE'], first_preds),2))
	a2 = str(round(calcAccuracy(data['COLOR_ROUTE'], freq_preds),2))
	a3 = str(round(calcAccuracy(data['COLOR_ROUTE'], hovered_preds),2))
	mydf.loc['ACCURACY'] = ['',a1,a2,a3]

	mydf = mydf.style.set_caption("Prediction of route used color")
	display(mydf)
	return mydf



