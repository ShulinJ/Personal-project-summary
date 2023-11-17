import lmdb
def writeCache(env, cache):
  with env.begin(write=True) as txn:  # 建立事务
    for k, v in cache.items():
      if type(k) == str:
        k = k.encode()
      if type(v) == str:
        v = v.encode()
      txn.put(k, v)  # 写入数据
file_A_path=r"C:\PROJECT\data\OCR_text_recognition\zhongyingwenhunhe\360+10+21+100xy40"
file_B_path=r"C:\PROJECT\data\OCR_text_recognition\zhongyingwenhunhe\NIPS2014"
file_C_path=r"C:\PROJECT\data\OCR_text_recognition\zhongyingwenhunhe\test_all"
out_path=r"C:\PROJECT\data\OCR_text_recognition\zhongyingwenhunhe\cn_en_ocr"
env_out = lmdb.open(out_path,int(5e10))#52428800
txn_out = env_out.begin(write=True)
cache = {}
path_list=[file_A_path,file_B_path,file_C_path]
a=0
with open("char_std_5990yuanshi.txt", 'rb') as file:
  DICT = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
for file_path in path_list:
  env = lmdb.Environment(file_path)
  txn = env.begin()
  for i in range(int(txn.get("num-samples".encode()).decode())):
    imageKey = 'image-%09d' % (i+1)
    labelKey = 'label-%09d' % (i+1)
    image=txn.get(imageKey.encode())
    if file_path==r"F:\data\360zw_ocr":
      list1 = txn.get(labelKey.encode()).decode().split(" ")
      label="".join(DICT[int(i)] for i in list1)

    else:
      label=txn.get(labelKey.encode())


    imageKey = 'image-%09d' % (a+1)
    labelKey = 'label-%09d' % (a+1)
    cache[imageKey] = image
    cache[labelKey] = label
    if i % 1000 == 0:
      writeCache(env_out, cache)
      cache = {}
      print("SOLVE", i)
    a+=1
  env.close()
cache['num-samples'] = str(a)
writeCache(env_out, cache)

env_out.close()
