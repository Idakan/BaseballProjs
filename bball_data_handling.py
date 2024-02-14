
# import packages needed for analysis


import pandas as pd
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

all_columns = ["Date", "Number of Game", "Day of Week", "Visiting Team", "Visting Team League",
     "Visiting Team Game Number", "Home Team", "Home Team League", "Home Team Game Number",
     "Visiting Score", "Home Score", "Length of Game in Outs", "Day or Night", "Completion Information",
     "Forfeit Information", "Protest Information", "Park ID", "Attendance", "Time of Game (minutes)",
     "Visiting Line Score", "Home Line Score",
     "Visiting At-Bats", "Visiting Hits", "Visiting Doubles", "Visiting Triples", "Visiting Home Runs",
     "Visiting RBI", "Visiting Sacrifice Hits", "Visiting Sacrifice Flies", "Visiting Hit-by-Pitch",
     "Visiting Walks", "Visiting Intentional Walks", "Visiting Strikeouts", "Visiting Stolen Bases",
     "Visiting Caught Stealing", "Visiting Grounded into Double Plays",
     "Visiting Awarded First on Catcher's Interference", "Visiting Left on Base",
     'Visiting Pitchers Used', 'Visiting Individual Earned Runs', 'Visiting Team Earned Runs',
     'Visiting Wild Pitches'
     , 'Visiting Balks', 'Visiting Putouts', 'Visiting Assists', 'Visiting Errors',
     'Visiting Passed Balls', 'Visiting Double Plays', 'Visiting Triple Plays', "Home At-Bats",
     "Home Hits", "Home Doubles", "Home Triples", "Home Home Runs", "Home RBI", "Home Sacrifice Hits",
     "Home Sacrifice Flies", "Home Hit-by-Pitch", "Home Walks", "Home Intentional Walks",
     "Home Strikeouts", "Home Stolen Bases", "Home Caught Stealing", "Home Grounded into Double Plays",
     "Home Awarded First on Catcher's Interference", "Home Left on Base", 'Home Pitchers Used',
     'Home Individual Earned Runs', 'Home Team Earned Runs', 'Home Wild Pitches', 'Home Balks',
     'Home Putouts', 'Home Assists', 'Home Errors', 'Home Passed Balls', 'Home Double Plays',
     'Home Triple Plays', 'Home Plate Umpire ID', 'Home Plate Umpire Name', '1B Umpire ID',
     '1B Umpire Name', '2B Umpire ID', '2B Umpire Name', '3B Umpire ID', '3B Umpire Name',
     'LF Umpire ID', 'LF Umpire Name', 'RF Umpire ID', 'RF Umpire Name', 'Visiting Manager ID',
     'Visiting Manager Name', 'Home Manager ID', 'Home Manager Name', 'Winning Pitcher ID',
     'Winning Pitcher Name', 'Losing Pitcher ID', 'Losing Pitcher Name', 'Saving Pitcher ID',
     'Saving Pitcher Name', 'Game Winning RBI Batter ID', 'Game Winning RBI Batter Name',
     'Visiting Starting Pitcher ID', 'Visiting Starting Pitcher Name', 'Home Starting Pitcher ID',
     'Home Starting Pitcher Name', 'Visiting Player 1 ID', 'Visiting Player 1 Name',
     'Visiting Player 1 Position', 'Visiting Player 2 ID', 'Visiting Player 2 Name',
     'Visiting Player 2 Position', 'Visiting Player 3 ID', 'Visiting Player 3 Name',
     'Visiting Player 3 Position', 'Visiting Player 4 ID', 'Visiting Player 4 Name',
     'Visiting Player 4 Position', 'Visiting Player 5 ID', 'Visiting Player 5 Name',
     'Visiting Player 5 Position', 'Visiting Player 6 ID', 'Visiting Player 6 Name',
     'Visiting Player 6 Position', 'Visiting Player 7 ID', 'Visiting Player 7 Name',
     'Visiting Player 7 Position', 'Visiting Player 8 ID', 'Visiting Player 8 Name',
     'Visiting Player 8 Position', 'Visiting Player 9 ID', 'Visiting Player 9 Name',
     'Visiting Player 9 Position', 'Home Player 1 ID', 'Home Player 1 Name', 'Home Player 1 Position',
     'Home Player 2 ID', 'Home Player 2 Name', 'Home Player 2 Position', 'Home Player 3 ID',
     'Home Player 3 Name', 'Home Player 3 Position', 'Home Player 4 ID', 'Home Player 4 Name',
     'Home Player 4 Position', 'Home Player 5 ID', 'Home Player 5 Name', 'Home Player 5 Position',
     'Home Player 6 ID', 'Home Player 6 Name', 'Home Player 6 Position', 'Home Player 7 ID',
     'Home Player 7 Name', 'Home Player 7 Position', 'Home Player 8 ID', 'Home Player 8 Name',
     'Home Player 8 Position', 'Home Player 9 ID', 'Home Player 9 Name', 'Home Player 9 Position',
     'Additional Information', 'Acquisition Information']
for num in range(1990, 2023):
  exec(f"df_{num} = pd.read_csv('C:/Users/ynakadi/PycharmProjects/General Projects/Lib/retrosheets/gl{num}.txt')")
  exec(f"df_{num}.columns = all_columns")
  exec(f"df_{num}['Date'] =  pd.to_datetime(df_{num} ['Date'], format='%Y%m%d')")
  exec(f"df_{num}['Game Type'] =  'Regular Season' ")


for playoff_series in ["wc", "dv", "lc", "ws"]:
  exec(f"df_{playoff_series} = pd.read_csv('C:/Users/ynakadi/PycharmProjects/General Projects/Lib/retrosheets/gl{playoff_series}.txt')")
  exec(f"df_{playoff_series}.columns = all_columns")
  exec(f"df_{playoff_series}['Date'] =  pd.to_datetime(df_{playoff_series} ['Date'], format='%Y%m%d')")
  exec(f"df_{playoff_series}['Game Type'] =  'Playoffs' ")


# Compose list of desired dataframes
list_df = []
exec("list_df = ["+ ",".join(['df_'+ str(i) for i in range(1990,2023)]) +"]")
print('HI')
for df in [df_wc, df_dv, df_lc, df_ws]:
    list_df.append(df)
df_total = pd.concat(list_df) # concatenate
df_total = pd.DataFrame(df_total) # convert to pandas DataFrame
df_total.shape # check the shape of large dataframe

# Exclude games with ties - https://www.espn.com/mlb/recap/_/gameId/220815115
df_total = df_total[df_total['Home Score'] != df_total['Visiting Score']]
df_total['Year'] = df_total['Date'].apply(lambda x: x.year)
df_total['Month'] = df_total['Date'].apply(lambda x: x.month)
pd.set_option('display.max_columns', None)

df_total.head(1)

# Determine "Home Result" and "Visiting Result"
loss_penalty = 0
df_total['Home Result'] = df_total.apply(lambda x: 1 if x['Visiting Score'] < x['Home Score'] else loss_penalty, axis=1)
df_total['Home Opponent Score'] = df_total['Visiting Score']
df_total['Home Opponent Line Score'] = df_total['Visiting Line Score']
df_total['Home Differential'] = df_total['Home Score'] - df_total['Visiting Score']

df_total['Visiting Result'] = df_total.apply(lambda x: 1 if x['Visiting Score'] > x['Home Score'] else loss_penalty, axis=1)
df_total['Visiting Opponent Score'] = df_total['Home Score']
df_total['Visiting Differential'] = -1*df_total['Home Score'] + df_total['Visiting Score']
df_total['Visiting Opponent Line Score'] = df_total['Home Line Score']

# Define home team columns
home_features = ["Date", "Home Team", "Home Team League", "Home Score", "Length of Game in Outs",
                 "Completion Information", "Forfeit Information", "Protest Information", "Home Line Score", "Home At-Bats",
                 "Home Hits", "Home Doubles", "Home Triples", "Home Home Runs", "Home RBI", "Home Sacrifice Hits",
                 "Home Sacrifice Flies", "Home Hit-by-Pitch", "Home Walks", "Home Intentional Walks", "Home Strikeouts",
                 "Home Stolen Bases", "Home Caught Stealing", "Home Grounded into Double Plays", "Home Awarded First on Catcher's Interference",
                 "Home Left on Base", 'Home Pitchers Used', 'Home Individual Earned Runs', 'Home Team Earned Runs', 'Home Wild Pitches', 'Home Balks', 'Home Putouts',
                 'Home Assists', 'Home Errors', 'Home Passed Balls', 'Home Double Plays', 'Home Triple Plays',
                 'Additional Information', 'Acquisition Information', 'Home Result', 'Home Team Game Number','Home Differential', 'Year','Month','Home Opponent Line Score', 'Home Opponent Score', "Game Type"]

# Define visiting team columns
visiting_features = ["Date", "Visiting Team", "Visting Team League", "Visiting Score", "Length of Game in Outs",
                 "Completion Information", "Forfeit Information", "Protest Information", "Visiting Line Score", "Visiting At-Bats",
                 "Visiting Hits", "Visiting Doubles", "Visiting Triples", "Visiting Home Runs", "Visiting RBI", "Visiting Sacrifice Hits",
                 "Visiting Sacrifice Flies", "Visiting Hit-by-Pitch", "Visiting Walks", "Visiting Intentional Walks", "Visiting Strikeouts",
                 "Visiting Stolen Bases", "Visiting Caught Stealing", "Visiting Grounded into Double Plays", "Visiting Awarded First on Catcher's Interference",
                 "Visiting Left on Base", 'Visiting Pitchers Used', 'Visiting Individual Earned Runs', 'Visiting Team Earned Runs', 'Visiting Wild Pitches', 'Visiting Balks', 'Visiting Putouts',
                 'Visiting Assists', 'Visiting Errors', 'Visiting Passed Balls', 'Visiting Double Plays', 'Visiting Triple Plays',
                 'Additional Information', 'Acquisition Information', 'Visiting Result', 'Visiting Team Game Number','Visiting Differential', 'Year','Month','Visiting Opponent Line Score', 'Visiting Opponent Score', "Game Type"]

# Define cumulative dataset column headers
columns = ["Date", "Team", "League", "Score", "Length of Game in Outs",
                 "Completion Information", "Forfeit Information", "Protest Information", "Line Score", "At-Bats",
                 "Hits", "Doubles", "Triples", "Home Runs", "RBI", "Sacrifice Hits",
                 "Sacrifice Flies", "Hit-by-Pitch", "Walks", "Intentional Walks", "Strikeouts",
                 "Stolen Bases", "Caught Stealing", "Grounded into Double Plays", "Awarded First on Catcher's Interference",
                 "Left on Base", 'Pitchers Used', 'Individual Earned Runs', 'Team Earned Runs', 'Wild Pitches', 'Balks', 'Putouts',
                 'Assists', 'Errors', 'Passed Balls', 'Double Plays', 'Triple Plays',
                 'Additional Information', 'Acquisition Information', "Result", 'Team Game Number', 'Differential', 'Year','Month','Opponent Line Score', 'Opponent Score', "Game Type"]

# Create home and visiting dataframes
df_visiting = df_total[visiting_features]
df_visiting.columns = columns

df_home = df_total[home_features]
df_home.columns = columns

# concatenate home and visiting dataframes into cumulative dataset
df_final = pd.concat([df_home, df_visiting])

df_final['Team_Fin'] = df_final['Year'].apply(lambda x: str(x)) + '_' + df_final['Team']
df_final.sort_values(by = ['Team', 'Date'])


team_names_df = pd.read_csv('C:/Users/ynakadi/PycharmProjects/General Projects/Lib/bballfiles/CurrentNames.csv')
team_names_df#['Team Name'] = team_names_df['CITY'] + ' ' + team_names_df['NICKNAME']
name_change_dict = team_names_df[['Former Name','Current Name']].set_index('Former Name').to_dict()['Current Name']
df_final["Team"] = df_final["Team"].apply(lambda x: name_change_dict[x])
team_names_df["Team Name"] = team_names_df['City'] + ' ' + team_names_df['Team']
team_names_df
#
team_dict = team_names_df[team_names_df['Former Name'].isin(name_change_dict.values())][['Former Name','Team Name']].set_index('Former Name').to_dict()['Team Name']
# name_change_dict
df_final["Team Name"] = df_final["Team"].apply(lambda x: team_dict[x])
len(set(df_final["Team Name"]))


team_color_map={
                "Arizona Diamondbacks" : '#940818',
                "Atlanta Braves":'#CF1010',
                "New York Yankees": "darkblue",
                "Houston Astros": "orange",
                "Baltimore Orioles": '#ff7f0e',
                "Cincinatti Reds":"#C6011F",
                "Los Angeles Dodgers": "#005A9C",
                "Texas Rangers" : "blue",
                "Tampa Bay Rays":"navy",
                "Toronto Blue Jays":"blue",
                "St. Louis Cardinals":"#B72126",
                "Washington Nationals":"#AB0003",
                "Pittsburgh Pirates":"#FDB827",
                "San Diego Padres": "#FFC425",
                "Seattle Mariners": "#005C5C",
                "San Francisco Giants":"#FD5A1E",
                "Oakland Athletics":"#003831",
                "New York Mets":"#002D72",
                "Minnesota Twins":"#002B5C",
                "Milwaukee Brewers":"#FFC52F",
                "Miami Marlins":"#00A3E0",
                "Kansas City Royals":"#004687",
                "Detroit Tigers":"#0C2340",
                "Colorado Rockies":"#333366",
                "Cleveland Indians":"#E50022",
                "Philadelphia Phillies": "red",
                "Chicago White Sox":"#27251F",
                "Chicago Cubs":"#0E3386",
                "Boston Red Sox":"#BD3039",
                "Los Angeles Angels":"#BA0021"

                }
data = df_final[df_final["Game Type"] == 'Regular Season'].sort_values(by=["Team","Date"])
teams = data["Team"].sort_values().unique()

