from pydub import AudioSegment
from pydub.utils import make_chunks
import os, re
# #
# # 迴圈目錄下所有檔案
for each in os.listdir("C:/Users/p8803/Desktop/天擎20210420/待切割"): #迴圈目錄
    
    filename = re.findall(r"(.*?)\.wav", each) # 取出.mp3字尾的檔名
    print(each)
    if each:
        # filename[0] += '.wav'
        # print(filename[0])

        mp3 = AudioSegment.from_file('C:/Users/p8803/Desktop/天擎20210420/待切割/{}'.format(each), "wav") # 開啟mp3檔案
        mp3[17*1000+500:].export(filename[0], format="wav") #
        size = 10000  # 切割的毫秒數 10s=10000

        chunks = make_chunks(mp3, size)  # 將檔案切割為幾秒一塊

        for i, chunk in enumerate(chunks):

            chunk_name = "c{}-{}.wav".format(each.split(".")[0],i)
            print(chunk_name)
            chunk.export('C:/Users/p8803/Desktop/天擎20210420/天擎20210420-切割/{}'.format(chunk_name), format="wav")