import pandas as pd

kbodata = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

print("oss project 2 제출")
print("컴퓨터공학과 12215535 김민경\n")

print("step 01.\n")

#해당 year의 category에서 상위 10명 선수 출력하는 함수
def print_top10(df, year, category):
    top_players = df.nlargest(10, category)
    print(f"The top 10 player of {category} in {year}\n")
    print(f"{top_players[['batter_name', category]]}\n")

#중첩 for문으로 2015부터 2018년까지 해당 카테고리의 상위 10명 선수 출력
for yr in range(2015, 2019):
    year = kbodata[kbodata['year'] == yr]
    for cate in ['H', 'avg', 'HR', 'OBP']:
        print_top10(year, yr, cate)


print("step 02.\n")


year_2018 = kbodata[kbodata['year'] == 2018]

mapping = year_2018.groupby('cp')

for position, group in mapping:
    top_player = group.loc[group['war'].idxmax()]
    print(f"The player of {position} with the highest war in 2018:{top_player[['batter_name', 'war']]}")


print("\nstep 03.\n")

categories = kbodata[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()

correlation_salary = categories['salary']

highest_corr = correlation_salary.abs().nlargest(2).idxmin()

print(f"The highest correlation with salary : {highest_corr}")