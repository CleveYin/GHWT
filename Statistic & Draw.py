# -*- coding: utf-8 -*-
# @Author: eraer
# @Date:   2022-03-06 12:10:52
# @Last Modified by:   cleve
# @Last Modified time: 2023-09-29 08:55:41


import os
import xarray as xr
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import linregress
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.metrics import mean_squared_error

# from osgeo import gdal
# import rioxarray

# 创建文件夹
def Create_Dir(path_1):
	if os.path.exists(path_1):
		pass
	else:
		os.makedirs(path_1)

# 计算高温热浪平均值
def Sta_mean(inPath_NC_1, data_1, start_1, end_1, outPath_NC_1):
	# 计算历史和未来高温热浪平均值（结果为NC）
	filename_Demo_1 = os.listdir(inPath_NC_1)[0].rsplit("_", 1)[0]

	data_List_1 = []
	year_List_1 = []
	attri_List_1 = []
	value_List_1 = []

	dataset_List_1 = []
	for year_1 in range(start_1, end_1 + 1):
		filename_1 = filename_Demo_1 + '_' + str(year_1) + '.nc'
		print(filename_1)
		inPath_NC_2 = os.path.join(inPath_NC_1, filename_1)
		dataset_1 = xr.open_dataset(inPath_NC_2)

		for variable_1 in ["HWF", "HWD", "HWAD", "HWAT", "HWSD", "HWED"]:
			data_List_1.append(data_1)
			year_List_1.append(year_1)
			attri_List_1.append(variable_1)
			value_List_1.append(np.nanmean(dataset_1[variable_1].values))		

		dataset_List_1.append(dataset_1)
	dataset_Years_1 = xr.concat(dataset_List_1, dim = "time")
	dataset_Mean_1 = dataset_Years_1.mean(dim = "time", skipna = True)

	dataframe_1 = pd.DataFrame(columns = ["Data", "Year", "Attribute", "Value"])
	dataframe_1["Data"] = data_List_1
	dataframe_1["Year"] = year_List_1
	dataframe_1["Attribute"] = attri_List_1
	dataframe_1["Value"] = value_List_1

	# 数据重采样，减少表格数据量或统一空间分辨率
	gap_1 = 1 # 输出NC文件时设为1，输出Excel时设为2
	lon_ref_1 = np.arange(-180 + gap_1 / 2, 180 + gap_1 / 2, gap_1)
	if data_1 == 'ERA5':
		lon_ref_1 = np.arange(0 + gap_1 / 2, 360 + gap_1 / 2, gap_1)
	lat_ref_1 = np.arange(-90 + gap_1 / 2, 90 + gap_1 / 2, gap_1)
	dataset_Mean_1 = dataset_Mean_1.interp(lon = lon_ref_1)
	dataset_Mean_1 = dataset_Mean_1.interp(lat = lat_ref_1)
	dataset_Mean_1 = dataset_Mean_1.assign_coords(data = data_1)
	dataset_Mean_1 = dataset_Mean_1.expand_dims('data')

	# if data_1 not in ["GLDAS", "ERA5", "CRU-JRA"]:
	# 	dataset_Mean_1 = dataset_Mean_1.roll(lon = int(180 / 1))

	# 将数据写入表格
	data_List_2 = []
	attri_List_2 = []
	value_List_2 = []
	for variable_1 in ["HWF", "HWD", "HWAD", "HWAT", "HWSD", "HWED"]:
		values_1 = dataset_Mean_1[variable_1].values.flatten()
		values_2 = values_1[~np.isnan(values_1)].tolist()
		for value_1 in values_2:
			data_List_2.append(data_1)
			attri_List_2.append(variable_1)
			value_List_2.append(value_1)
	dataframe_2 = pd.DataFrame(columns = ["Data", "Attribute", "Value"])
	dataframe_2["Data"] = data_List_2
	dataframe_2["Attribute"] = attri_List_2
	dataframe_2["Value"] = value_List_2
	dataset_Mean_1.to_netcdf(os.path.join(outPath_NC_1, filename_Demo_1 + "_mean_" + str(start_1) + "_" + str(end_1) + ".nc"))
	print(data_1 + " is done!")
	return dataframe_1, dataframe_2

# 计算单个像元的斜率和P值
def Cal_trend(y_1):
	x_1 = np.arange(len(y_1))	# 自变量
	nan_count_1 = np.count_nonzero(np.isnan(y_1))	# 计算因变量中空值的数量
	if nan_count_1 >= 0 and nan_count_1 <= len(y_1) * 2 / 5:	# 如果因变量中空值的数量少于三分之一，则删除空值计算斜率
		nan_pos_1 = np.argwhere(np.isnan(y_1))	# 因变量中空值的索引
		nan_pos_2 = nan_pos_1.flatten().tolist()
		x_1 = np.delete(x_1, nan_pos_2)	# 删除空值位置上对应的自变量
		y_1 = y_1[~np.isnan(y_1)]	# 删除因变量中的空值
	result = linregress(x_1, y_1)
	slope = result.slope
	pvalue = result.pvalue
	pvalue_1 = np.nan
	pvalue_2 = np.nan
	if pvalue <= 0.05:
		pvalue_1 = np.nan
		pvalue_2 = np.nan
	if pvalue > 0.05:
		if slope < 0:
			pvalue_1 = -1
		elif slope >= 0:
			pvalue_2 = 1
		slope = np.nan
	return slope, pvalue_1, pvalue_2

# 计算多年高温热浪的斜率和P值
def Sta_slope(inPath_NC_1, data_1, start_1, end_1, outPath_NC_1):
	filename_Demo_1 = os.listdir(inPath_NC_1)[0].rsplit("_", 1)[0]
	dataset_List_1 = []
	for year_1 in range(start_1, end_1 + 1):
		filename_1 = filename_Demo_1 + '_' + str(year_1) + '.nc'
		print(filename_1)
		inPath_NC_2 = os.path.join(inPath_NC_1, filename_1)
		dataset_1 = xr.open_dataset(inPath_NC_2)
		dataset_List_1.append(dataset_1)
	dataset_Years_1 = xr.concat(dataset_List_1, dim = "time")

	gap_1 = 1
	lon_ref_1 = np.arange(-180 + gap_1 / 2, 180 + gap_1 / 2, gap_1)
	if data_1 == 'ERA5':
		lon_ref_1 = np.arange(0 + gap_1 / 2, 360 + gap_1 / 2, gap_1)
	lat_ref_1 = np.arange(-90 + gap_1 / 2, 90 + gap_1 / 2, gap_1)
	dataset_Years_1 = dataset_Years_1.interp(lon = lon_ref_1)
	dataset_Years_1 = dataset_Years_1.interp(lat = lat_ref_1)

	lon_1 = dataset_Years_1.lon.values
	lat_1 = dataset_Years_1.lat.values
	dataset_List_2 = []
	for attri_1 in ["HWF", "HWD", "HWAD", "HWAT", "HWSD", "HWED"]:
		data_2 = dataset_Years_1[attri_1].data
		results_1 = np.apply_along_axis(Cal_trend, 0, data_2)
		# plt.imshow(results_1[0])
		# plt.show()
		dataset_Result_1 = xr.Dataset({attri_1 + "_slope": (["lat", "lon"], results_1[0]), attri_1 + "_pValue_1": (["lat", "lon"], results_1[1]), attri_1 + "_pValue_2": (["lat", "lon"], results_1[2])}, coords = {"lon": (["lon"], lon_1), "lat": (["lat"], lat_1)})
		dataset_List_2.append(dataset_Result_1)
		print(attri_1 + " is done!")
	dataset_HWE_1 = xr.merge(dataset_List_2)
	dataset_HWE_1.to_netcdf(os.path.join(outPath_NC_1, filename_Demo_1 + "_slope_" + str(start_1) + "_" + str(end_1) + ".nc"))

# 将经度范围从[0, 360]调整到[-180, 180]
def Dataset_Roll(dataset_1):
	# 读取原始坐标数据
	lat_1 = dataset_1['lat'].data
	lon_1 = dataset_1['lon'].data
	time_1 = dataset_1['time'].data

	# 将数据沿180°经线拆分，准备拼接
	gap_1 = lon_1[1] - lon_1[0]
	dataset_2 = dataset_1.sel(lon = slice(0, 180))
	dataset_3 = dataset_1.sel(lon = slice(180 + gap_1, 360))

	# 数据拼接
	datasetList_1 = []
	varList_1 = list(dataset_1.keys())
	for var_1 in varList_1:
		dataArray_1 = np.append(dataset_3[var_1].data, dataset_2[var_1].data, axis = 1)

		# 构建新经度
		lon_2 = []
		for index_1 in range(dataArray_1.shape[1]):
			lon_2.append(-180 + gap_1 * index_1)

		dataset_4 = xr.Dataset({var_1: (['time', 'lat', 'lon'], [dataArray_1])}, coords = {'time': (['time'], time_1), 'lat': (['lat'], lat_1), 'lon': (['lon'], lon_2)})
		datasetList_1.append(dataset_4)

	dataset_5 = xr.merge(datasetList_1)
	return dataset_5

# 将历史高温热浪统一空间分辨率，并计算逐年平均值
def Resam_Mean_Historic(inPath_NC_1, outPath_1):
	gap_1 = 1
	lon_ref_1 = np.arange(-180 + gap_1 / 2, 180 + gap_1 / 2, gap_1)
	lat_ref_1 = np.arange(-90 + gap_1 / 2, 90 + gap_1 / 2, gap_1)

	for year_1 in range(1971, 2021):
		dataset_Year_List_1 = []
		for data_1 in ["GLDAS", "ERA5", "CRU-JRA"]:
			inPath_NC_2 = os.path.join(inPath_NC_1, data_1, '3_HWE_Annual', data_1.lower() + '_air_temp_daily_tmax_35_3_HWE_' + str(year_1) + '.nc')
			dataset_1 = xr.open_dataset(inPath_NC_2)
			if data_1 == "ERA5":
				dataset_1 = Dataset_Roll(dataset_1)
			dataset_1 = dataset_1.interp(lon = lon_ref_1)
			dataset_1 = dataset_1.interp(lat = lat_ref_1)
			dataset_1 = dataset_1.assign_coords(data = data_1)
			dataset_1 = dataset_1.expand_dims('data')
			dataset_Year_List_1.append(dataset_1)
			print(year_1, data_1)

		dataset_Year_1 = xr.concat(dataset_Year_List_1, dim = "data")
		dataset_Year_1 = dataset_Year_1.mean(dim = "data", skipna = True)
		outPath_NC_2 = os.path.join(outPath_NC_1, 'historical_air_temp_daily_tmax_35_3_HWE_' + str(year_1) + '.nc')
		dataset_Year_1.to_netcdf(outPath_NC_2)

# 各模式未来热浪数据-历史平均热浪数据
def Future_Historical(inPath_NC_1, inPath_NC_2, outPath_1):
	dataframe_1 = pd.DataFrame(columns = ["GCM", "Year", "Statistic", "Value"])
	model_List_1 = []
	year_List_1 = []
	stati_List_1 = []
	value_List_1 = []

	dataframe_2 = pd.DataFrame(columns = ["GCM", "Statistic", "Value"])
	model_List_2 = []
	stati_List_2 = []
	value_List_2 = []

	gap_1 = 1
	lon_ref_1 = np.arange(-180 + gap_1 / 2, 180 + gap_1 / 2, gap_1)
	lat_ref_1 = np.arange(-90 + gap_1 / 2, 90 + gap_1 / 2, gap_1)

	model_List_3 = ['access_cm2', 'awi_cm_1_1_mr', 'cmcc_esm2', 'cnrm_cm6_1', 'cnrm_esm2_1', 'ec_earth3_veg_lr', \
	'gfdl_esm4', 'inm_cm4_8', 'inm_cm5_0', 'kiost_esm', 'miroc6', 'miroc_es2l', 'mpi_esm1_2_lr', 'mri_esm2_0', 'noresm2_mm']
	for model_1 in model_List_3:
		for year_1 in range(1971, 2015):
			inPath_NC_3 = os.path.join(inPath_NC_1, 'historical_air_temp_daily_tmax_35_3_HWE_' + str(year_1) + '.nc')
			dataset_1 = xr.open_dataset(inPath_NC_3)
			inPath_NC_4 = os.path.join(inPath_NC_2, model_1 + '_historical', '3_HWE_Annual', model_1 + '_historical_air_temp_daily_tmax_35_3_HWE_' + str(year_1) + '.nc')
			dataset_2 = xr.open_dataset(inPath_NC_4)
			dataset_2 = Dataset_Roll(dataset_2)
			dataset_2 = dataset_2.interp(lon = lon_ref_1)
			dataset_2 = dataset_2.interp(lat = lat_ref_1)

			# 计算检测出的热浪像元数量占热浪像元总数量的比例
			dataset_3 = xr.where(dataset_1.notnull() & dataset_2.notnull(), 1, np.nan)
			perc_1 = np.count_nonzero(1 - np.isnan(dataset_3['HWD'])) / np.count_nonzero(1 - np.isnan(dataset_1['HWD']))

			model_List_1.append(model_1.replace('_', '-').upper())
			year_List_1.append(year_1)
			stati_List_1.append('Recognition rate')
			value_List_1.append(perc_1 * 100)

			# 计算三个热浪属性的RMSE
			mask_1 = ~np.isnan(dataset_3['HWD'].data[:, :, 0])			
			for var_1 in ['HWF', 'HWD']:
				rmse_1 = mean_squared_error(dataset_1[var_1].data[:, :, 0][mask_1], dataset_2[var_1].data[0, :, :][mask_1], squared = False)

				model_List_1.append(model_1.replace('_', '-').upper())
				year_List_1.append(year_1)
				stati_List_1.append('RMSE (' + var_1 + ')')
				value_List_1.append(rmse_1)

			print(model_1, year_1)

	dataframe_1["GCM"] = model_List_1
	dataframe_1["Year"] = year_List_1
	dataframe_1["Statistic"] = stati_List_1
	dataframe_1["Value"] = value_List_1

	for model_2 in model_List_3:
		for var_1 in ['Recognition rate', 'RMSE (HWF)', 'RMSE (HWD)']:
			dataframe_3 = dataframe_1[(dataframe_1["GCM"] == model_2.replace('_', '-').upper()) & (dataframe_1["Statistic"] == var_1)] 
			model_List_2.append(model_2.replace('_', '-').upper())
			stati_List_2.append(var_1)
			value_List_2.append(dataframe_3['Value'].mean())

	dataframe_2["GCM"] = model_List_2
	dataframe_2["Statistic"] = stati_List_2
	dataframe_2["Value"] = value_List_2

	outPath_2 = os.path.join(outPath_1, 'future-historical_HWE_summary.csv')
	outPath_3 = os.path.join(outPath_1, 'future-historical_HWE_mean.csv')
	dataframe_1.to_csv(outPath_2, index = False)
	dataframe_2.to_csv(outPath_3, index = False)

# 将未来高温热浪统一空间分辨率，并计算各模式逐年平均值，并输出逐年平均值表格和各模式平均值表格
def Resam_Mean_Future(inPath_NC_1, outPath_1):
	dataframe_1 = pd.DataFrame(columns = ["Scenario", "GCM", "Year", "Attribute", "Value"])
	scenario_List_1 = []
	model_List_2 = []
	year_List_1 = []
	attri_List_1 = []
	value_List_1 = []

	gap_1 = 1
	lon_ref_1 = np.arange(-180 + gap_1 / 2, 180 + gap_1 / 2, gap_1)
	lat_ref_1 = np.arange(-90 + gap_1 / 2, 90 + gap_1 / 2, gap_1)

	model_List_1 = ['access_cm2', 'awi_cm_1_1_mr', 'cmcc_esm2', 'cnrm_cm6_1', 'cnrm_esm2_1', 'ec_earth3_veg_lr', \
	'gfdl_esm4', 'inm_cm4_8', 'inm_cm5_0', 'kiost_esm', 'miroc6', 'miroc_es2l', 'mpi_esm1_2_lr', 'mri_esm2_0', 'noresm2_mm']

	for experiment_1 in ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]:
		for year_1 in range(2015, 2101):
			dataset_Year_List_1 = []
			for model_1 in model_List_1:
				inPath_NC_2 = os.path.join(inPath_NC_1, model_1 + '_' + experiment_1, '3_HWE_Annual', model_1 + '_' + experiment_1 + '_air_temp_daily_tmax_35_3_HWE_' + str(year_1) + '.nc')
				dataset_1 = xr.open_dataset(inPath_NC_2)
				dataset_1 = Dataset_Roll(dataset_1)
				dataset_1 = dataset_1.interp(lon = lon_ref_1)
				dataset_1 = dataset_1.interp(lat = lat_ref_1)
				dataset_1 = dataset_1.assign_coords(gcm = model_1)
				dataset_1 = dataset_1.expand_dims('gcm')
				dataset_Year_List_1.append(dataset_1)

				for variable_1 in ["HWF", "HWD", "HWAD", "HWAT", "HWSD", "HWED"]:
					scenario_List_1.append(experiment_1.replace('_', ' ', 1).replace("_", '.', 1).upper())
					model_List_2.append(model_1.replace('_', '-').upper())
					year_List_1.append(year_1)
					attri_List_1.append(variable_1)
					value_List_1.append(np.nanmean(dataset_1[variable_1].values))		
				print(experiment_1, year_1, model_1)	

			dataset_Year_1 = xr.concat(dataset_Year_List_1, dim = "gcm")
			dataset_Year_1 = dataset_Year_1.mean(dim = "gcm", skipna = True)
			outPath_2 = os.path.join(outPath_1, 'yearly_mean', experiment_1)
			Create_Dir(os.path.join(outPath_1, 'yearly_mean', experiment_1))
			outPath_3 = os.path.join(outPath_2, "gcms-" + experiment_1 + '_air_temp_daily_tmax_35_3_HWE_' + str(year_1) + '.nc')
			# dataset_Year_1.to_netcdf(outPath_3)

	dataframe_1["Scenario"] = scenario_List_1
	dataframe_1["GCM"] = model_List_2
	dataframe_1["Year"] = year_List_1
	dataframe_1["Attribute"] = attri_List_1 
	dataframe_1["Value"] = value_List_1 

	outPath_4 = os.path.join(outPath_1, "future_yearly_summary.csv")
	dataframe_1.to_csv(outPath_4, index = False)	

# 计算暴露度
def Exposure(inPath_HWE_1, inPath_soc_1, tag_1, outPath_1):
	dataframe_1 = pd.DataFrame(columns = ["Society", "Scenario", "Year", "HWD", "Exposure"])
	soci_List_1 = []
	scen_List_1 = []
	year_List_1 = []
	HWD_List_1 = []
	expo_List_1 = []
	index_1 = 0
	for model_1 in ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]:
		for year_1 in range(2030, 2101, 10):
			dataset_List_1 = []
			for year_2 in range(year_1 - 9, year_1 + 1):
				filename_1 = 'gcms-' + model_1 + '_air_temp_daily_tmax_35_3_HWE_' + str(year_2) + '.nc'
				inPath_HWE_2 = os.path.join(inPath_HWE_1, model_1, filename_1)
				dataset_HWE_1 = xr.open_dataset(inPath_HWE_2)
				dataset_List_1.append(dataset_HWE_1)
			dataset_HWE_2 = xr.concat(dataset_List_1, dim = "time")
			dataset_HWE_2 = dataset_HWE_2.mean(dim = "time", skipna = True)

			filename_soc_1 = model_1.split("_")[0] + "_" + str(year_1)
			if tag_1 == 'GDP':
				filename_soc_1 = model_1.split("_")[0].upper() + "_gdp" + str(year_1) + '.tif'
			inPath_soc_2 = os.path.join(inPath_soc_1, filename_soc_1 + ".nc")
			dataset_soc_1 = xr.open_dataset(inPath_soc_2)

			# 统一空间分辨率
			lat_2 = dataset_soc_1["lat"].values
			lon_2 = dataset_soc_1["lon"].values
			dataset_HWE_2 = dataset_HWE_2.interp(lat = lat_2) # specify calculation
			dataset_HWE_2 = dataset_HWE_2.interp(lon = lon_2) # specify calculation

			# 计算暴露度
			array_hwd_1 = dataset_HWE_2["HWD"]
			dataset_expo_1 = array_hwd_1 * dataset_soc_1
			dataset_expo_2 = dataset_soc_1.where(array_hwd_1 > 1)

			if tag_1 == 'Population':
				variable_1 = filename_soc_1
			if tag_1 == 'GDP':
				variable_1 = "gdp"
			exposure_1 = dataset_expo_1[variable_1]
			exposure_1.rio.set_spatial_dims("lon", "lat", inplace = True)
			exposure_1.rio.set_crs("epsg:4326", inplace = True)
			exposure_2 = dataset_expo_2[variable_1]
			exposure_2.rio.set_spatial_dims("lon", "lat", inplace = True)
			exposure_2.rio.set_crs("epsg:4326", inplace = True)

			outPath_2 = os.path.join(outPath_1, tag_1, "exposure")
			Create_Dir(outPath_2)
			outPath_3 = os.path.join(outPath_1, tag_1, "count")
			Create_Dir(outPath_3)
			outPath_4 = os.path.join(outPath_2, model_1 + "_HWD-" + tag_1 + "_exposure_" + str(year_1) + ".tif")
			exposure_1.rio.to_raster(outPath_4)
			outPath_5 = os.path.join(outPath_3, model_1 + "_HWD-" + tag_1 + "_count_" + str(year_1) + ".tif")
			exposure_2.rio.to_raster(outPath_5)

			class_ref_1 = [2, 30, 60, 90, 10000]
			for index_2 in range(len(class_ref_1) - 1):
				array_hwd_2 = xr.where((array_hwd_1 >= class_ref_1[index_2]) & (array_hwd_1 < class_ref_1[index_2 + 1]), 1, np.nan)
				dataset_soc_2 = dataset_soc_1.where(array_hwd_2 == 1)
				expo_sum_1 = np.nansum(dataset_soc_2[variable_1].values)

				if tag_1 == 'Population':
					expo_sum_1 = np.nansum(dataset_soc_2[variable_1].values) / 1000000000
					tag_2 = tag_1 + " (billion)"
				if tag_1 == 'GDP':
					expo_sum_1 = np.nansum(dataset_soc_2[variable_1].values) / 1000000000000
					tag_2 = tag_1 + " (trillion)"

				HWD_1 = str(class_ref_1[index_2] + 1) + '-' + str(class_ref_1[index_2 + 1])
				if index_2 == 3:
					HWD_1 = ">91"

				soci_List_1.append(tag_2)
				scen_List_1.append(model_1.replace('_', ' ', 1).replace("_", '.', 1).upper())
				year_List_1.append(year_1)
				HWD_List_1.append(HWD_1)
				expo_List_1.append(expo_sum_1)
			print(str(year_1) + " of " + model_1 + " is done!")
		index_1 = index_1 + 1
	dataframe_1["Society"] = soci_List_1
	dataframe_1["Scenario"] = scen_List_1
	dataframe_1["Year"] = year_List_1
	dataframe_1["HWD"] = HWD_List_1
	dataframe_1["Exposure"] = expo_List_1
	return dataframe_1

# 计算风险
def Risk(inPath_HWE_1, inPath_pop_1, inPath_gdp_1, outPath_1):
	HWE_all_max_1 = 0
	pop_all_max_1 = 0
	gdp_all_max_1 = 0
	risk_all_min_1 = 0
	risk_all_max_1 = 0
	for model_1 in ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]:
		dataset_List_1 = []
		for year_1 in [2030, 2070]:
			dataset_List_2 = []
			for year_2 in range(year_1, year_1 + 31, 10):
				# 计算热浪十年均值
				dataset_List_3 = []
				for year_3 in range(year_2 - 9, year_2 + 1):
					filename_1 = 'gcms-' + model_1 + '_air_temp_daily_tmax_35_3_HWE_' + str(year_3) + '.nc'
					inPath_HWE_2 = os.path.join(inPath_HWE_1, model_1, filename_1)
					dataset_HWE_1 = xr.open_dataset(inPath_HWE_2)
					dataset_List_3.append(dataset_HWE_1)
					print(year_3)
				dataset_HWE_2 = xr.concat(dataset_List_3, dim = "time")
				dataset_HWE_2 = dataset_HWE_2.mean(dim = "time", skipna = True)

				# 读取相应年代人口数据
				filename_pop_1 = model_1.split("_")[0] + "_" + str(year_2)
				inPath_pop_2 = os.path.join(inPath_pop_1, filename_pop_1 + ".nc")
				dataset_pop_1 = xr.open_dataset(inPath_pop_2)

				# 读取相应年代GDP数据
				filename_gdp_1 = model_1.split("_")[0].upper() + "_gdp" + str(year_2) + '.tif'
				inPath_gdp_2 = os.path.join(inPath_gdp_1, filename_gdp_1 + ".nc")
				dataset_gdp_1 = xr.open_dataset(inPath_gdp_2)

				# 统一空间分辨率
				lat_1 = dataset_pop_1["lat"].values
				lon_1 = dataset_pop_1["lon"].values
				dataset_HWE_2 = dataset_HWE_2.interp(lat = lat_1) # specify calculation
				dataset_HWE_2 = dataset_HWE_2.interp(lon = lon_1) # specify calculation
				dataset_gdp_1 = dataset_gdp_1.interp(lat = lat_1) # specify calculation
				dataset_gdp_1 = dataset_gdp_1.interp(lon = lon_1) # specify calculation

				HWE_max_1 = np.nanmax(dataset_HWE_2['HWD'].data)
				pop_max_1 = np.nanmax(dataset_pop_1[filename_pop_1].data)
				gdp_max_1 = np.nanmax(dataset_gdp_1['gdp'].data)

				# 确定整体最大值和最小值
				if HWE_max_1 > HWE_all_max_1:
					HWE_all_max_1 = HWE_max_1 
				if pop_max_1 > pop_all_max_1:
					pop_all_max_1 = pop_max_1 
				if gdp_max_1 > gdp_all_max_1:
					gdp_all_max_1 = gdp_max_1 

				dataset_HWE_2 = dataset_HWE_2 / (282.62 - 3)
				dataset_pop_1 = dataset_pop_1 / 10011008
				dataset_gdp_1 = 1 - dataset_gdp_1 / 253123624783.56

				HWE_array_1 = dataset_HWE_2['HWD'].data
				pop_array_1 = dataset_pop_1[filename_pop_1].data
				gdp_array_1 = dataset_gdp_1['gdp'].data

				risk_array_1 = (HWE_array_1 + pop_array_1 + gdp_array_1) / 3

				time_1 = pd.date_range(str(year_2) + "-01-01", periods = 1)
				dataset_risk_1 = xr.Dataset({str(year_1) + "-" + str(year_1 + 30): (["time", "lat", "lon"], [risk_array_1])}, coords = {"time": time_1, "lat": (["lat"], lat_1), "lon": (["lon"], lon_1)})
				dataset_List_2.append(dataset_risk_1)
			dataset_risk_2 = xr.concat(dataset_List_2, dim = "time")
			dataset_risk_3 = dataset_risk_2.mean(dim = "time", skipna = True)
			risk_min_1 = np.nanmin(dataset_risk_3[str(year_1) + "-" + str(year_1 + 30)].data)
			risk_max_1 = np.nanmax(dataset_risk_3[str(year_1) + "-" + str(year_1 + 30)].data)
			if risk_min_1 < risk_all_min_1:
				risk_all_min_1 = risk_min_1 
			if risk_max_1 > risk_all_max_1:
				risk_all_max_1 = risk_max_1 
			dataset_risk_3 = (dataset_risk_3 - 0.2359) / (0.6936 - 0.2359)
			dataset_List_1.append(dataset_risk_3)
		dataset_risk_4 = xr.merge(dataset_List_1)
		dataset_risk_4.to_netcdf(os.path.join(outPath_1, "risk_" + model_1 + ".nc"))
	print(HWE_all_max_1, pop_all_max_1, gdp_all_max_1, risk_min_1, risk_max_1)

# 将NetCDF数据转换为TIFF
def NCToTIFF(inPath_1, attri_1, outPath_AT_1):
	dataset_1 = xr.open_dataset(inPath_1)
	dataarray_1 = dataset_1[attri_1]
	if attri_1 == "HWF":
		class_ref_1 = [5, 10, 15]
		dataarray_1 = xr.where((dataarray_1 >= 1) & (dataarray_1 < class_ref_1[0]), int(1), dataarray_1)
		dataarray_1 = xr.where((dataarray_1 >= class_ref_1[0]) & (dataarray_1 < class_ref_1[1]), int(2), dataarray_1)
		dataarray_1 = xr.where((dataarray_1 >= class_ref_1[1]) & (dataarray_1 < class_ref_1[2]), int(3), dataarray_1)
		dataarray_1 = xr.where(dataarray_1 >= class_ref_1[2], 4, dataarray_1)
	elif attri_1 == "HWD":
		class_ref_1 = [30, 60, 90]
		dataarray_1 = xr.where((dataarray_1 >= 3) & (dataarray_1 < class_ref_1[0]), int(1), dataarray_1)
		dataarray_1 = xr.where((dataarray_1 >= class_ref_1[0]) & (dataarray_1 < class_ref_1[1]), int(2), dataarray_1)
		dataarray_1 = xr.where((dataarray_1 >= class_ref_1[1]) & (dataarray_1 < class_ref_1[2]), int(3), dataarray_1)
		dataarray_1 = xr.where(dataarray_1 >= class_ref_1[2], 4, dataarray_1)
	dataarray_1.rio.set_spatial_dims("lon", "lat", inplace = True)
	dataarray_1.rio.set_crs("epsg:4326", inplace = True)
	dataarray_1.rio.to_raster(outPath_AT_1)

# 将TIFF数据转换为NetCDF
def TIFFToNC(inPathList_1, society_1, SSP_1, outPath_1):
	lon_1 = np.linspace(-180, 180, 2880)
	lat_1 = np.linspace(-55.88, 83.85, 1117)
	if society_1 == "HWD-GDP_exposure_":
		lon_1 = np.linspace(-180, 180, 4320)
		lat_1 = np.linspace(-90, 90, 2160)
	dataset_List_1 = []
	for inPath_1 in inPathList_1:
		raster_1 = gdal.Open(inPath_1)
		array_1 = raster_1.ReadAsArray()
		array_1[array_1 < 0] = np.nan
		array_1 = np.flipud(array_1)
		year_1 = inPath_1.split('\\')[-1].split('.tif')[0].split('exposure_')[1]
		dataset_1 = xr.Dataset({year_1: (["lat", "lon"], array_1)}, coords = {"lon": (["lon"], lon_1), "lat": (["lat"], lat_1)})
		dataset_List_1.append(dataset_1)
	dataset_2 = xr.merge(dataset_List_1)
	# dataset_2.rio.set_spatial_dims('lon', 'lat', inplace = True)
	# dataset_2 = dataset_2.rio.write_crs("epsg:4326", inplace = True)
	# dataset_2 = dataset_2.rio.clip_box(minx = -180, miny = -58, maxx = 180, maxy = 85)
	dataset_2.to_netcdf(os.path.join(outPath_1, SSP_1 + society_1[0: -1] + ".nc"))

# 绘制热浪平均值（三个子图）
def Draw_Raster_1(inPath_NC_List_1, attri_1, title_1, color_1, label_List_1, outPath_fig_1):
	projection = ccrs.PlateCarree()
	axes_class = (GeoAxes, dict(map_projection = projection))
	fig = plt.figure(figsize = (21, 3))
	plt.subplots_adjust(left = 0.115, bottom = 0.06, right = 0.96, top = 0.95)
	axgr = AxesGrid(fig, 111, axes_class = axes_class, nrows_ncols = (1, 3), axes_pad = 0.05, cbar_location = "right", \
	cbar_mode = "single", cbar_pad = 0.1, cbar_size = "2%", label_mode = "")  # note the empty label_mode
	ax_1 = fig.add_axes([0.03, 0.12, 0.08, 0.77])	# [x, y, width, height]
	color_list_1 = ["#1f77b4", "#ff7f0e", "#2ca02c"]
	# 计算最大值和最小值
	dataset_1 = xr.open_dataset(inPath_NC_List_1[0])
	data_1 = dataset_1[attri_1].data
	dataset_2 = xr.open_dataset(inPath_NC_List_1[1])
	data_2 = dataset_2[attri_1].data
	dataset_3 = xr.open_dataset(inPath_NC_List_1[2])
	data_3 = dataset_3[attri_1].data
	min_value_1 = np.min([np.nanmin(data_1), np.nanmin(data_2), np.nanmin(data_3)])
	max_value_1 = np.max([np.nanmax(data_1), np.nanmax(data_2), np.nanmax(data_3)])
	print(min_value_1, max_value_1) 
	print(np.nanmean(data_1), np.nanmean(data_2), np.nanmean(data_3)) 
	color_2 = matplotlib.cm.get_cmap(color_1, lut = len(label_List_1) - 1)
	color_list_2 = []
	for index_1 in range(color_2.N):
		color_list_2.append(matplotlib.colors.rgb2hex(color_2(index_1)))
	index_2 = 0
	for i, ax in enumerate(axgr):
		ax.set_extent([-179.99, 180, -58, 85])
		dataset_4 = xr.open_dataset(inPath_NC_List_1[index_2])
		data_4 = dataset_4[attri_1].data
		lat_1 = dataset_4["lat"].data
		lon_1 = dataset_4["lon"].data
		try:
			mean_lat_1 = np.nanmean(data_4[0], axis = 1)
		except Exception as e:
			mean_lat_1 = np.nanmean(data_4, axis = 1)
		ax_1.plot(mean_lat_1, lat_1, color = color_list_1[index_2])
		ax_1.set_yticks([-58, -30, 0, 30, 60, 85])
		# ax_1.set_ylabels([])
		ax_1.tick_params(labelsize = 18)
		ax_1.spines["right"].set_visible(False)
		ax_1.spines["top"].set_visible(False)
		p = ax.pcolor(lon_1, lat_1, data_4, cmap = color_2, vmin = label_List_1[0], vmax = label_List_1[-1])	# 绘制风险使用
		# try:
		# 	p = ax.pcolor(lon_1, lat_1, data_4[0], cmap = color_2, vmin = label_List_1[0], vmax = label_List_1[-1])
		# except Exception as e:
		# 	p = ax.pcolor(lon_1, lat_1, data_4, cmap = color_2, norm = matplotlib.colors.LogNorm(vmin = label_List_1[0], vmax = label_List_1[-1]))
		ax.coastlines()
		count_list_1 = []
		for index_3 in range(1, len(label_List_1)):
			data_5 = np.where((data_4 >= label_List_1[index_3 - 1]) & (data_4 < label_List_1[index_3]), 1, 0)
			count_1 = np.count_nonzero(data_5)
			count_list_1.append(count_1)
		if index_2 == 0:
			ax_2 = fig.add_axes([0.11, -0.03, 0.08, 0.77])
			ax_2.pie(count_list_1, radius = 0.8, wedgeprops = {"width": 0.3, "edgecolor":"k","linewidth": 0.6}, colors = color_list_2)
		elif index_2 == 1:
			ax_2 = fig.add_axes([0.39, -0.03, 0.08, 0.77])
			ax_2.pie(count_list_1, radius = 0.8, wedgeprops = {"width": 0.3, "edgecolor":"k","linewidth": 0.6}, colors = color_list_2)
		else:
			ax_2 = fig.add_axes([0.67, -0.03, 0.08, 0.77])
			ax_2.pie(count_list_1, radius = 0.8, wedgeprops = {"width": 0.3, "edgecolor":"k","linewidth": 0.6}, colors = color_list_2)			
		index_2 = index_2 + 1
	clobar = axgr.cbar_axes[0].colorbar(p)
	# del label_List_1[1: -1: 2]
	clobar.set_ticks(label_List_1)
	if attri_1 == "HWSD" or attri_1 == "HWED":
		label_List_1 = ["1/1", "3/1", "5/1", "7/1", "9/1", "11/1", "12/31"]
	if (attri_1 in ["2030-2060", "2070-2100"]) and ('risk' not in inPath_NC_List_1[0]):
		label_List_2 = []
		for label_1 in label_List_1:
			if label_1 in [1, 10]:
				label_List_2.append(label_1)
			else:
				label_List_2.append("10$^{" + str(int(np.log10(label_1))) + "}$")
		label_List_1 = label_List_2
	clobar.set_ticklabels(label_List_1)
	clobar.ax.tick_params(labelsize = 18)
	# plt.text(title_1, fontsize = 20, ha = "center", va = "bottom")
	plt.text(-2.32, 1.03, title_1, fontsize = 20, transform = ax.transAxes)
	# plt.show()
	# plt.savefig(outPath_fig_1)

# 绘制热浪变化率（三个子图）
def Draw_Raster_2(inPath_NC_List_1, attri_1, title_1, color_1, label_List_1, outPath_fig_1):
	projection = ccrs.PlateCarree()
	axes_class = (GeoAxes, dict(map_projection = projection))
	fig = plt.figure(figsize = (21, 3))
	plt.subplots_adjust(left = 0.115, bottom = 0.06, right = 0.96, top = 0.95)
	axgr = AxesGrid(fig, 111, axes_class = axes_class, nrows_ncols = (1, 3), axes_pad = 0.05, cbar_location = "right", \
	cbar_mode = "single", cbar_pad = 0.1, cbar_size = "2%", label_mode = "")  # note the empty label_mode
	ax_1 = fig.add_axes([0.03, 0.12, 0.08, 0.77])	# [x, y, width, height]
	color_list_1 = ["#1f77b4", "#ff7f0e", "#2ca02c"]
	# 计算最大值和最小值
	dataset_1 = xr.open_dataset(inPath_NC_List_1[0])
	data_1 = dataset_1[attri_1 + "_slope"].data
	dataset_2 = xr.open_dataset(inPath_NC_List_1[1])
	data_2 = dataset_2[attri_1 + "_slope"].data
	dataset_3 = xr.open_dataset(inPath_NC_List_1[2])
	data_3 = dataset_3[attri_1 + "_slope"].data
	min_value_1 = np.min([np.nanmin(data_1), np.nanmin(data_2), np.nanmin(data_3)])
	max_value_1 = np.max([np.nanmax(data_1), np.nanmax(data_2), np.nanmax(data_3)])
	# print(min_value_1, max_value_1) 
	print(np.nanmean(data_1), np.nanmean(data_2), np.nanmean(data_3)) 
	color_2 = matplotlib.cm.get_cmap(color_1, lut = len(label_List_1) - 1)
	color_list_2 = []
	for index_1 in range(color_2.N):
		color_list_2.append(matplotlib.colors.rgb2hex(color_2(index_1)))
	norm = matplotlib.colors.TwoSlopeNorm(vmin = min_value_1, vcenter = 0, vmax = max_value_1)
	index_2 = 0
	for i, ax in enumerate(axgr):
		ax.set_extent([-179.99, 180, -58, 85])
		dataset_4 = xr.open_dataset(inPath_NC_List_1[index_2])
		data_4 = dataset_4[attri_1 + "_slope"].data
		data_5 = dataset_4[attri_1 + "_pValue_1"].data
		data_6 = dataset_4[attri_1 + "_pValue_2"].data
		lat_1 = dataset_4["lat"].data
		lon_1 = dataset_4["lon"].data
		mean_lat_1 = np.nanmean(data_4, axis = 1)
		ax_1.plot(mean_lat_1, lat_1, color = color_list_1[index_2])
		ax_1.set_yticks([-58, -30, 0, 30, 60, 85])
		# ax_1.set_ylabels([])
		ax_1.tick_params(labelsize = 18)
		ax_1.spines["right"].set_visible(False)
		ax_1.spines["top"].set_visible(False)
		# p = ax.contourf(lon_1, lat_1, data_4, transform = projection, cmap = color_1, vmin = min_value_1, vmax = max_value_1)
		p = ax.pcolor(lon_1, lat_1, data_4, cmap = color_2, vmin = label_List_1[0], vmax = label_List_1[-1])
		ax.contourf(lon_1, lat_1, data_5, colors = "yellow")
		ax.contourf(lon_1, lat_1, data_6, colors = "green")
		ax.coastlines()
		count_list_1 = []
		for index_3 in range(1, len(label_List_1)):
			data_6 = np.where((data_4 >= label_List_1[index_3 - 1]) & (data_4 < label_List_1[index_3]), 1, 0)
			count_1 = np.count_nonzero(data_6)
			count_list_1.append(count_1)
		if index_2 == 0:
			ax_2 = fig.add_axes([0.11, -0.03, 0.08, 0.77])
			ax_2.pie(count_list_1, radius = 0.8, wedgeprops = {"width": 0.3, "edgecolor":"k","linewidth": 0.6}, colors = color_list_2)
		elif index_2 == 1:
			ax_2 = fig.add_axes([0.39, -0.03, 0.08, 0.77])
			ax_2.pie(count_list_1, radius = 0.8, wedgeprops = {"width": 0.3, "edgecolor":"k","linewidth": 0.6}, colors = color_list_2)
		else:
			ax_2 = fig.add_axes([0.67, -0.03, 0.08, 0.77])
			ax_2.pie(count_list_1, radius = 0.8, wedgeprops = {"width": 0.3, "edgecolor":"k","linewidth": 0.6}, colors = color_list_2)	
		index_2 = index_2 + 1
	clobar = axgr.cbar_axes[0].colorbar(p)
	clobar.set_ticks(label_List_1)
	if attri_1 in ["2030-2060", "2070-2100"]:
		label_List_2 = []
		for label_1 in label_List_1:
			if label_1 in [1, 10]:
				label_List_2.append(label_1)
			else:
				label_List_2.append("10$^{" + str(int(np.log10(label_1))) + "}$")
		label_List_1 = label_List_2
	clobar.set_ticklabels(label_List_1)
	clobar.ax.tick_params(labelsize = 18)
	# plt.text(title_1, fontsize = 20, ha = "center", va = "bottom")
	plt.text(-2.32, 1.03, title_1, fontsize = 20, transform = ax.transAxes)
	# plt.show()
	plt.savefig(outPath_fig_1, dpi = 300)

def main():
	# # Figure 3-5：计算历史高温热浪多年平均值和变化率
	# outPath_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary"
	# dataframe_List_1 = []
	# dataframe_List_2 = []
	# for data_1 in ["GLDAS", "ERA5", "CRU-JRA"]:
	# 	inPath_NC_1 = os.path.join(r"I:\1_papers\6_heat wave variation\2_data\1_reanalyze_climate", data_1, "3_HWE_Annual")
	# 	dataframe_1, dataframe_2 = Sta_mean(inPath_NC_1, data_1, 1971, 2020, outPath_1)
	# 	dataframe_List_1.append(dataframe_1)
	# 	dataframe_List_2.append(dataframe_2)
	# 	Sta_slope(inPath_NC_1, data_1, 1971, 2020, outPath_1)
	# dataframe_3 = pd.concat(dataframe_List_1)
	# dataframe_4 = pd.concat(dataframe_List_2)
	# outPath_excel_1 = os.path.join(outPath_1, "historical_hwe_yearly_1971-2020.csv")
	# dataframe_3.to_csv(outPath_excel_1, index = False)
	# outPath_excel_2 = os.path.join(outPath_1, "historical_hwe_mean_1971-2020.csv")
	# dataframe_4.to_csv(outPath_excel_2, index = False)

	# # Figure 6：将历史高温热浪统一空间分辨率，并计算逐年平均值
	# inPath_NC_1 = r'I:\1_papers\6_heat wave variation\2_data\1_reanalyze_climate'
	# outPath_NC_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary\1_yearly mean'
	# Resam_Mean_Historic(inPath_NC_1, outPath_NC_1)

	# # Figure 6：计算各GCM历史数据计算的热浪与历史数据的差
	# inPath_NC_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary\1_yearly mean'
	# inPath_NC_2 = r'I:\1_papers\6_heat wave variation\2_data\2_CMIP6_cliamte\3_result'
	# outPath_NC_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary'
	# Future_Historical(inPath_NC_1, inPath_NC_2, outPath_NC_1)

	# # Figure 9：将未来高温热浪统一空间分辨率，并计算各模式逐年平均值，并输出逐年平均值表格和各模式平均值表格
	# inPath_NC_1 = r'I:\1_papers\6_heat wave variation\2_data\2_CMIP6_cliamte\3_result'
	# outPath_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary'
	# Resam_Mean_Future(inPath_NC_1, outPath_1)

	# # Figure 7-8：计算未来高温热浪多年平均值和变化率
	# outPath_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary"
	# dataframe_List_1 = []
	# dataframe_List_2 = []
	# for data_1 in ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]:
	# 	inPath_NC_1 = os.path.join(r"I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\yearly_mean", data_1)
	# 	dataframe_1, dataframe_2 = Sta_mean(inPath_NC_1, data_1, 2015, 2100, outPath_1)
	# 	dataframe_List_1.append(dataframe_1)
	# 	dataframe_List_2.append(dataframe_2)
	# 	Sta_slope(inPath_NC_1, data_1, 2015, 2100, outPath_1)
	# dataframe_3 = pd.concat(dataframe_List_1)
	# dataframe_4 = pd.concat(dataframe_List_2)
	# outPath_excel_1 = os.path.join(outPath_1, "modeled_hwe_yearly_2015-2100.csv")
	# dataframe_3.to_csv(outPath_excel_1, index = False)
	# outPath_excel_2 = os.path.join(outPath_1, "modeled_hwe_mean_2015-2100.csv")
	# dataframe_4.to_csv(outPath_excel_2, index = False)

	# # 计算人口和GDP暴露度
	# inPath_HWE_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\yearly_mean'
	# inPath_List_1 = [r'I:\1_papers\6_heat wave variation\2_data\3_society\population', r"I:\1_papers\6_heat wave variation\2_data\3_society\gdp"]
	# tag_List_1 = ['Population', "GDP"]
	# outPath_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure"
	# index_1 = 0
	# dataframe_List_1 = []
	# for inPath_1 in inPath_List_1:
	# 	dataframe_1 = Exposure(inPath_HWE_1, inPath_1, tag_List_1[index_1], outPath_1)
	# 	dataframe_List_1.append(dataframe_1)
	# 	index_1 = index_1 + 1
	# dataframe_2 = pd.concat(dataframe_List_1)
	# outPath_2 = os.path.join(outPath_1, "HWD_exposure.csv")
	# dataframe_2.to_csv(outPath_2, index = False)

	# # 计算风险
	# inPath_HWE_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\yearly_mean'
	# inPath_pop_1 = r'I:\1_papers\6_heat wave variation\2_data\3_society\population'
	# inPath_gdp_1 = r"I:\1_papers\6_heat wave variation\2_data\3_society\gdp"
	# outPath_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\risk"
	# Risk(inPath_HWE_1, inPath_pop_1, inPath_gdp_1, outPath_1)

	# 绘制多年高温热浪平均值和变化率
	order_1 = 0
	task_1 = "未来斜率"
	if task_1 == "当代平均":
		order_1 = 3
		title_1 = "_CRU-JRA-ERA5-GLDAS_"
		inPath_NC_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary\cru-jra_air_temp_daily_tmax_35_3_HWE_mean_1971_2020.nc"
		inPath_NC_2 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary\era5_air_temp_daily_tmax_35_3_HWE_mean_1971_2020.nc"
		inPath_NC_3 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary\gldas_air_temp_daily_tmax_35_3_HWE_mean_1971_2020.nc"
		title_list_1 = ["HWF (times)", "HWD (days)", "HWAD (days)", "HWAT (°C)", "HWSD (date)", "HWED (date)"]
		label_List_1 = [[1, 3, 5, 7, 9, 11, 13, 15], 
						[3, 15, 27, 39, 51, 63, 75, 87],
						[3, 6, 9, 12, 15, 18, 21, 24], 
						[35, 37, 39, 41, 43, 45, 47],
						[1, 59, 120, 181, 243, 304, 366], 
						[1, 59, 120, 181, 243, 304, 366]]
	elif task_1 == "当代斜率":
		order_1 = 4
		title_1 = "_CRU-JRA-ERA5-GLDAS_"
		inPath_NC_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary\cru-jra_air_temp_daily_tmax_35_3_HWE_slope_1971_2020.nc"
		inPath_NC_2 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary\era5_air_temp_daily_tmax_35_3_HWE_slope_1971_2020.nc"
		inPath_NC_3 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\1_historical_summary\gldas_air_temp_daily_tmax_35_3_HWE_slope_1971_2020.nc"
		title_list_1 = ["HWF (times / a)", "HWD (days / a)", "HWAD (days / a)", "HWAT (°C / a)", "HWSD (days / a)", "HWED (days / a)"]	
		label_List_1 = [[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45], 
						[-3, -2, -1, 0, 1, 2, 3],
						[-0.18, -0.12, -0.06, 0, 0.06, 0.12, 0.18],
						[-0.21, -0.14, -0.07, 0, 0.07, 0.14, 0.21],
						[-8.7, -5.8, -2.9, 0, 2.9, 5.8, 8.7], 
						[-10.2, -6.8, -3.4, 0, 3.4, 6.8, 10.2]]
	elif task_1 == "未来平均":
		order_1 = 7
		title_1 = "_GCMs_"
		inPath_NC_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\gcms-ssp1_2_6_air_temp_daily_tmax_35_3_HWE_mean_2015_2100.nc"
		inPath_NC_2 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\gcms-ssp2_4_5_air_temp_daily_tmax_35_3_HWE_mean_2015_2100.nc"
		inPath_NC_3 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\gcms-ssp5_8_5_air_temp_daily_tmax_35_3_HWE_mean_2015_2100.nc"	
		title_list_1 = ["HWF (times)", "HWD (days)", "HWAD (days)", "HWAT (°C)", "HWSD (date)", "HWED (date)"]
		label_List_1 = [[1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 
						[3, 42, 81, 120, 159, 198, 237],
						[3, 14, 25, 36, 47, 58, 69, 80],
						[35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5],
						[1, 59, 120, 181, 243, 304, 366], 
						[1, 59, 120, 181, 243, 304, 366]]
	elif task_1 == "未来斜率":
		order_1 = 8
		title_1 = "_GCMs_"
		inPath_NC_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\gcms-ssp1_2_6_air_temp_daily_tmax_35_3_HWE_slope_2015_2100.nc"
		inPath_NC_2 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\gcms-ssp2_4_5_air_temp_daily_tmax_35_3_HWE_slope_2015_2100.nc"
		inPath_NC_3 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\2_modeled_summary\gcms-ssp5_8_5_air_temp_daily_tmax_35_3_HWE_slope_2015_2100.nc"	
		title_list_1 = ["HWF (times / a)", "HWD (days / a)", "HWAD (days / a)", "HWAT (°C / a)", "HWSD (days / a)", "HWED (days / a)"]	
		label_List_1 = [[-0.21, -0.14, -0.07, 0, 0.07, 0.14, 0.21], 
						[-3.6, -2.4, -1.2, 0, 1.2, 2.4, 3.6],
						[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],
						[-0.18, -0.12, -0.06, 0, 0.06, 0.12, 0.18],
						[-3.3, -2.2, -1.1, 0, 1.1, 2.2, 3.3], 
						[-3, -2, -1, 0, 1, 2, 3]]
	inPath_NC_List_1 = [inPath_NC_1, inPath_NC_2, inPath_NC_3]
	attri_list_1 = ["HWF", "HWD", "HWAD", "HWAT", "HWSD", "HWED"]
	color_list_1 = ["inferno_r", "inferno_r", "inferno_r", "inferno_r", "viridis_r", "viridis_r"]
	# color_list_1 = [cmaps.GMT_hot_r, cmaps.GMT_hot_r, cmaps.GMT_hot_r, cmaps.GMT_panoply, cmaps.GMT_panoply]
	outPath_fig_1 = r"I:\1_papers\6_heat wave variation\3_figure\3_Figure\source"
	index_1 = 1
	for index_2 in range(6):
		attri_1 = attri_list_1[index_2]
		title_2 = title_list_1[index_2]
		color_1 = color_list_1[index_2]
		outPath_fig_2 = os.path.join(outPath_fig_1, "figure " + str(order_1) + "_" + str(index_1 + index_2) + title_1 + attri_1 + ".pdf")
		# Draw_Raster_1(inPath_NC_List_1, attri_1, title_2, color_1, label_List_1[index_2], outPath_fig_2)
		Draw_Raster_2(inPath_NC_List_1, attri_1, title_2, "RdBu_r", label_List_1[index_2], outPath_fig_2)

	# # 将TIFF数据转换为NetCDF
	# inPath_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\exposure'
	# outPath_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\netcdf'
	# for society_1 in ["HWD-Population_exposure_", "HWD-GDP_exposure_"]:
	# 	for SSP_1 in ["ssp1_2_6_", "ssp2_4_5_", "ssp5_8_5_"]:
	# 		inPathList_1 = []
	# 		for year_1 in ["2030-2060", "2070-2100"]:
	# 			filename_1 = SSP_1 + society_1 + year_1 + ".tif"
	# 			inPath_2 = os.path.join(inPath_1, filename_1)
	# 			inPathList_1.append(inPath_2)
	# 		TIFFToNC(inPathList_1, society_1, SSP_1, outPath_1)

	# # 绘制人口和GDP暴露度
	# # 降低数据分辨率
	# inPath_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\netcdf'
	# inPath_1 = r'I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\risk'	
	# filenameList_1 = os.listdir(inPath_1)
	# gap_1 = 1
	# for filename_1 in filenameList_1:
	# 	inPath_2 = os.path.join(inPath_1, filename_1)
	# 	dataset_1 = xr.open_dataset(inPath_2)

	# 	lat_1 = dataset_1['lat'].data
	# 	lon_ref_1 = np.arange(-180 + gap_1 / 2, 180 + gap_1 / 2, gap_1)
	# 	lat_ref_1 = np.arange(lat_1[0] + gap_1 / 2, lat_1[-1] + gap_1 / 2, gap_1)

	# 	dataset_1 = dataset_1.interp(lat = lon_ref_1) # specify calculation
	# 	dataset_1 = dataset_1.interp(lon = lon_ref_1) # specify calculation
	# 	outPath_1 = os.path.join(inPath_1, filename_1.split('.nc')[0] + '_res.nc')
	# 	dataset_1.to_netcdf(outPath_1)
	# task_1 = "GDP暴露度"
	# if task_1 == "人口暴露度":
	# 	order_1 = 12
	# 	title_1 = "_pop_exposure_"
	# 	inPath_NC_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\netcdf\ssp1_2_6_HWD-Population_exposure_res.nc"
	# 	inPath_NC_2 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\netcdf\ssp2_4_5_HWD-Population_exposure_res.nc"
	# 	inPath_NC_3 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\netcdf\ssp5_8_5_HWD-Population_exposure_res.nc"
	# 	title_list_1 = ["Population exposure (person·day, 2030-2060)", "Population exposure (person·day, 2070-2100)"]	
	# 	label_List_1 = [[10, 1e+02, 1e+03, 1e+04, 1e+05, 1e+06, 1e+7, 1e+8, 1e+9],
	# 					[10, 1e+02, 1e+03, 1e+04, 1e+05, 1e+06, 1e+7, 1e+8, 1e+9]]
	# 	color_list_1 = ["PRGn_r", "PRGn_r"]
	# elif task_1 == "GDP暴露度":
	# 	order_1 = 12
	# 	title_1 = "_GDP_exposure_"
	# 	inPath_NC_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\netcdf\ssp1_2_6_HWD-GDP_exposure_res.nc"
	# 	inPath_NC_2 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\netcdf\ssp2_4_5_HWD-GDP_exposure_res.nc"
	# 	inPath_NC_3 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\mean\netcdf\ssp5_8_5_HWD-GDP_exposure_res.nc"	
	# 	title_list_1 = ["GDP exposure (dollar·day, 2030-2060)", "GDP exposure (dollar·day, 2070-2100)"]	
	# 	label_List_1 = [[1e+02, 1e+04, 1e+06, 1e+08, 1e+10, 1e+12, 1e+14], 
	# 					[1e+02, 1e+04, 1e+06, 1e+08, 1e+10, 1e+12, 1e+14]]
	# 	color_list_1 = ["PRGn_r", "PRGn_r"]
	# inPath_NC_List_1 = [inPath_NC_1, inPath_NC_2, inPath_NC_3]
	# attri_list_1 = ["2030-2060", "2070-2100"]
	# outPath_fig_1 = r"I:\1_papers\6_heat wave variation\3_figure\3_Figure\source"
	# index_1 = 1
	# for index_2 in range(2):
	# 	attri_1 = attri_list_1[index_2]
	# 	title_2 = title_list_1[index_2]
	# 	color_1 = color_list_1[index_2]
	# 	outPath_fig_2 = os.path.join(outPath_fig_1, "figure " + str(order_1) + "_" + str(index_1 + index_2) + title_1 + attri_1 + ".pdf")
	# 	Draw_Raster_1(inPath_NC_List_1, attri_1, title_2, color_1, label_List_1[index_2], outPath_fig_2)

	# # 绘制高温热浪风险
	# order_1 = 12
	# title_1 = "_risk_"
	# inPath_NC_1 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\risk\risk_ssp1_2_6_res.nc"
	# inPath_NC_2 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\risk\risk_ssp2_4_5_res.nc"
	# inPath_NC_3 = r"I:\1_papers\6_heat wave variation\2_data\7_statistic\3_exposure\risk\risk_ssp5_8_5_res.nc"	
	# title_list_1 = ["Heat wave risk (2030-2060)", "Heat wave risk (2070-2100)"]	
	# label_List_1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	# inPath_NC_List_1 = [inPath_NC_1, inPath_NC_2, inPath_NC_3]
	# attri_list_1 = ["2030-2060", "2070-2100"]
	# outPath_fig_1 = r"I:\1_papers\6_heat wave variation\3_figure\3_Figure\source"
	# index_1 = 1
	# for index_2 in range(2):
	# 	attri_1 = attri_list_1[index_2]
	# 	title_2 = title_list_1[index_2]
	# 	outPath_fig_2 = os.path.join(outPath_fig_1, "figure " + str(order_1) + "_" + str(index_1 + index_2) + title_1 + attri_1 + ".pdf")
	# 	Draw_Raster_1(inPath_NC_List_1, attri_1, title_2, "PRGn_r", label_List_1, outPath_fig_2)

	# # 将NetCDF数据转换为TIFF
	# model_1 = "GFDL-ESM4_ssp585"
	# inPath_1 = os.path.join(r"G:\1_papers\6_heat wave variation\2_data\CMIP6", model_1, r"3_HWE_Annual\1_HWE")
	# outPath_1 = r"G:\1_papers\6_heat wave variation\2_data\Statistic\4_exposure\1_HWE"
	# for dirPath, dirname, filenames in os.walk(inPath_1):
	# 	for filename in filenames:
	# 		year_1 = int(filename.split(".nc")[0].rsplit("_")[-1])
	# 		if year_1 % 10 == 0:
	# 			inPath_2 = os.path.join(inPath_1, filename)
	# 			for attri_1 in ["HWF", "HWD"]:
	# 				outPath_2 = os.path.join(outPath_1, model_1, attri_1)
	# 				Create_Dir(outPath_2)
	# 				outPath_3 = os.path.join(outPath_2, filename.split(".nc")[0] + "_" + attri_1 + ".tif")
	# 				NCToTIFF(inPath_2, attri_1, outPath_3)

main()


