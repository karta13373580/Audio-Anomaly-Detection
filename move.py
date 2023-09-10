import os, random, shutil

def moveFile(fileDir):
    pathDir = os.listdir(fileDir)    #取圖片的原始路徑
    filenumber = len(pathDir)
    rate=0.5    #自定義抽取圖片的比例，比方說100張抽10張，那就是0.1
    picknumber=int(filenumber*rate) #按照rate比例從資料夾中取一定數量圖片
    sample = random.sample(pathDir, picknumber)  #隨機選取picknumber數量的樣本圖片
    print (sample)
    for name in sample:
            shutil.move(fileDir+name, tarDir+name)
    return
 
if __name__ == '__main__':
    fileDir = "D:/AudioGANomaly_bohan/data/paper_mimii_dataset/valve/id04ttt/test/normal/"    #源圖片資料夾路徑
    tarDir = 'D:/AudioGANomaly_bohan/data/paper_mimii_dataset/valve/id04ttt/test_true/normal/'    #移動到新的資料夾路徑
    moveFile(fileDir)