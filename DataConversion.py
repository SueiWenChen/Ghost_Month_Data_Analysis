import pandas as pd

#%% load data
frames = []
for year in range(2014, 2021):
    df = pd.read_csv(str(year)+'.csv', encoding='utf-8')
    frames.append(df)
frames = pd.concat(frames).reset_index(drop=True)
frames = frames.drop(columns=["Location_of_drowning"])
#%% load entries for inspection/translation
# CityOrCounty_ = list(set(frames["City_or_County"]))
TypesOfWaters_ = list(set(frames["Types_of _waters"]))
DrowningReason_ = list(set(frames["Drowning_reasons"]))
DrowningResult_ = list(set(frames["Drowning_results"]))
Gender_ = list(set(frames["Gender"]))
Age_ = list(set(frames["Age"]))
SwimmingSkills_ = list(set(frames["Swimming_skills"]))

#%% English counterparts
municipality = {"臺北市": "Taipei City",
                "新北市":"New Taipei City",
                "桃園市":"Taoyuan City",
                "臺中市":"Taichung City",
                "臺中港":"Taichung Port",
                "臺南市":"Tainan City", 
                "高雄市":"Kaohsiung City",
                "高雄港":"Kaohsiung Harbor", 
                "基隆市":"Keelung City",
                "基隆港":"Keelung Harbor", 
                "新竹市":"Hsinchu City",
                "嘉義市":"Chiayi City", 
                "新竹縣":"Hsinchu County", 
                "苗栗縣":"Miaoli County",
                "彰化縣":"Changhua County",
                "南投縣":"Nantou County",
                "雲林縣":"Yunlin County",
                "嘉義縣":"Chiayi County",
                "屏東縣":"Pingtung County", 
                "宜蘭縣":"Yilan County",
                "花蓮縣":"Hualien County", 
                "臺東縣":"Taitung County", 
                "澎湖縣":"Penghu County",
                "金門縣":"Kinmin County",        
                "連江縣":"Lienchiang County"}


waters = {"碼頭":"Dock",
          "魚塭":"Fish Pond",
          "圳溝":"Ditch",
          "湖潭":"Lake",
          "池塘":"Pond",
          "水庫":"Reservoir",
          "近海(海岸線1公里內)":"Offshore (<1km)",
          "外海(海岸線1公里以外)":"Offshore (>1km)",
          "溪河":"River",
          "游泳池":"Swimming Pool",
          "其他":"Others"}

results = {"失蹤":"Missing", "獲救":"Rescued", "死亡":"Dead"}
gender = {"男":"Male", "女":"Female", "不詳":"Unknown"}
swim = {"會":"Yes", "不會":"No", "不詳":"Unknown"}

reasons = {"工作":"Work",
          "自殺":"Suicide",
          "浮屍":"Floating Corpse",
          "翻船":"Capsizing",
          "失足":"Slipping",
          "交通事故":"Traffic Accident",
          "戲水":"Playing",
          "潛水":"Snorkeling",
          "救人":"Saving Others",
          "垂釣":"Fishing",
          "其他":"Other"}

#%% Simplify the Drowning Reasons
N = frames.shape[0] # number of rows
for i in range(N):
    for reason in reasons.keys():
        if reason in frames.at[i,"Drowning_reasons"]:
            frames.at[i,"Drowning_reasons"] = reason
#%% translate to English
frames = frames.replace({"City_or_County": municipality , \
                         "Types_of _waters": waters, \
                        "Drowning_reasons": reasons, \
                        "Drowning_results": results, \
                        "Gender": gender, \
                        "Swimming_skills": swim} )
frames = frames.replace("不詳", "Unknown")
#%% save as csv to convert into chinese calendar using the formula
# TEXT(date_string, "[$-130000]yyyy-mm-dd")
frames.to_csv("DrowningData.csv", index = False)


#%%
# After using Excel to convert the dates into the Chinese calendar, we need to 
# take care of the 閏月's:
# 2014: 10/24-11/21 閏九月
# 2017: 07/23-08/21 閏六月
# 2020: 05/23-06/20 閏四月
#%%

data = pd.read_csv('DrowningData.csv', encoding='utf-8')
N = data.shape[0]
# get dates of chinese calendar
ccDate = data["CC_Date"]
ccYear = [int(ccDate[i][:4]) for i in range(N)]
ccMonth = [int(ccDate[i][5:7]) for i in range(N)]
ccDay = [int(ccDate[i][8:]) for i in range(N)]

#%%
# loop over each date to see if conversion is needed
for i in range(N):
    # 2014
    if ccYear[i]==2014:
        if ccMonth[i]==10: # if date is during 2014 閏九月
            ccMonth[i]=0 # set the ccMonth to 0
        elif ccMonth[i]>10: # if date is after that month
            ccMonth[i] -= 1 
    # 2017
    if ccYear[i]==2017:
        if ccMonth[i]==7: # if date is during 2017 閏六月
            ccMonth[i]=0 # set the ccMonth to 0
        elif ccMonth[i]>7: # if date is after that month
            ccMonth[i] -= 1 
    # 2020
    if ccYear[i]==2020:
        if ccMonth[i]==5: # if date is during 2020 閏四月
            ccMonth[i]=0 # set the ccMonth to 0
        elif ccMonth[i]>5: # if date is after that month
            ccMonth[i] -= 1                     
    
#%%
# adding chinese calendar to data
data = data.drop("CC_Date", axis = 1)
#%%
data.insert(4, "CC_Year", ccYear, False)
data.insert(5, "CC_Month", ccMonth, False)
data.insert(6, "CC_Day", ccDay, False)
#%% save as csv
data.to_csv("DrowningData.csv", index = False)

