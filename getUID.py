from utils import *


# path = 'Data/02-06-2020/DICOM/Lung_Dx-G0011'
# path = '/home/wangshuo/Desktop/test_data/Lung-PET-CT-Dx/G0011'
# path = 'Data/02-06-2020/DICOM/Lung_Dx-G0011/04-29-2009-LUNGC-51228/2.000000-A phase 5mm Stnd SS50-53792/2-017.dcm'
# path = 'Lung-PET-CT-Dx/Lung_Dx-E0001/10-25-2007-NA-lung-83596/3.000000-5mm Lung SS50-36046/1-01.dcm'
# path = 'Lung-PET-CT-Dx/Lung_Dx-E0001/'
path = 'Lung-PET-CT-Dx/Lung_Dx-A0001/04-04-2007-NA-Chest-07990/2.000000-5mm-40805/'


def getUID_path(path):
    dict = {}
    list = os.listdir(path)
    print("path: ", path)
    print("List: ", list)

    for date in list:
        date_path = os.path.join(path, date)
        series_list = os.listdir(date_path)
        series_list.sort()
        print("series_list: ",series_list)

        for series in series_list:
            series_path = os.path.join(date_path, series)
            dicom_list = os.listdir(series_path)
            dicom_list.sort()

            for dicom in dicom_list:
                dicom_path = os.path.join(series_path, dicom)
                info = loadFileInformation(dicom_path)
                dict[info['dicom_num']] = (dicom_path, dicom)
    # print("dict: ", dict)
    return dict


def getUID_file(path):
    info = loadFileInformation(path)
    UID = info['dicom_num']
    print("UID: ", UID)
    return UID

# dict = getUID_file(path)
# print("dict: ",dict)