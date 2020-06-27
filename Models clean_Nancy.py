{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "import math\n",
    "import statsmodels.api as sm\n",
    "from statsmodels.tools.eval_measures import rmse\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from math import sqrt\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.model_selection import cross_val_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn import preprocessing\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this function should be applied to a df containing the data for matches (aggregated or not) in the initial \n",
    "# clean format that we defined\n",
    "# THIS FUNCTION ORDERS DISTANCES AND CLUSTERS FROM HIGHGEST TO LOWEST\n",
    "def aggData(df):\n",
    "    distances = df.iloc[:, 6:12]\n",
    "    distances.fillna(-999)\n",
    "    centroids = df.iloc[:, 12:16]\n",
    "    centroids.fillna(-999)\n",
    "    clusters = df.iloc[:, 16:20].values\n",
    "    new_distances = np.sort(distances.values, axis=1)[:,::-1]\n",
    "    new_centroids = np.sort(centroids.values, axis=1)[:,::-1]\n",
    "    new_clusters = np.sort(clusters, axis=1)[:,::-1]\n",
    "    distances = distances.replace(-999, np.nan)\n",
    "    centroids = centroids.replace(-999, np.nan)\n",
    "    df.iloc[:, 6:12] = new_distances\n",
    "    df.iloc[:, 12:16] = new_centroids\n",
    "    df.iloc[:, 16:20] = new_clusters\n",
    "    df['agility']=np.where(df['strategy']!=df['strategy'].shift(-1), 1,0)\n",
    "    ranking = df['ranking']\n",
    "    df = df.drop('ranking', axis=1)\n",
    "    df.insert(len(df.columns), 'ranking', ranking)\n",
    "    df['cluster_A'] = df['cluster_A'] / df['n_alive']\n",
    "    df['cluster_B'] = df['cluster_B'] / df['n_alive']\n",
    "    df['cluster_C'] = df['cluster_C'] / df['n_alive']\n",
    "    df['cluster_D'] = df['cluster_D'] / df['n_alive']\n",
    "    df = df[df.n_alive > 0]\n",
    "    df = df.drop(['time','n_alive','in_aircraft'], axis=1)\n",
    "    df.loc[:,['distance1_2','distance1_3','distance1_4','distance2_3', 'distance2_4', 'distance3_4',\n",
    "              'distance_centroid1', 'distance_centroid2', 'distance_centroid3', 'distance_centroid4']] = df.loc[:,['distance1_2','distance1_3','distance1_4','distance2_3', 'distance2_4', 'distance3_4',\n",
    "                                                                                                                   'distance_centroid1', 'distance_centroid2', 'distance_centroid3', 'distance_centroid4']] / 100\n",
    "    df.loc[:, ['distance1_2','distance1_3','distance1_4','distance2_3', 'distance2_4', 'distance3_4',\n",
    "              'distance_centroid1', 'distance_centroid2', 'distance_centroid3', 'distance_centroid4']] = df.loc[:, ['distance1_2','distance1_3','distance1_4','distance2_3', 'distance2_4', 'distance3_4',\n",
    "                                                                                                                   'distance_centroid1', 'distance_centroid2', 'distance_centroid3', 'distance_centroid4']].applymap(lambda x:np.log(x) if x != 0 else 0)\n",
    "    means = df.groupby(['matchId', 'teamId']).mean().reset_index()\n",
    "    stds = df.groupby(['matchId', 'teamId']).std().reset_index()\n",
    "    means = means.fillna(99)\n",
    "    means = means.replace(np.inf, 0)\n",
    "    stds = stds.fillna(0)\n",
    "    new_df = pd.merge(stds, means, how = 'left', on=['matchId', 'teamId'])\n",
    "    new_df = new_df.drop(['n_players_x', 'ranking_x', 'agility_x'], axis=1)\n",
    "    return new_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('combined_csv_5.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:33: RuntimeWarning: invalid value encountered in log\n"
     ]
    }
   ],
   "source": [
    "aggregated_df = aggData(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df3 = pd.read_csv('combined_csv_4.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "//anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:33: RuntimeWarning: invalid value encountered in log\n"
     ]
    }
   ],
   "source": [
    "aggregated_df3 = aggData(df3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2 = pd.read_csv('combined_csv_2.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "//anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:33: RuntimeWarning: invalid value encountered in log\n"
     ]
    }
   ],
   "source": [
    "aggregated_df2 = aggData(df2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'aggregated_df' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-4-dae793b0128d>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdf_123\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconcat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0maggregated_df\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maggregated_df2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0maggregated_df3\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mignore_index\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'aggregated_df' is not defined"
     ]
    }
   ],
   "source": [
    "df_123 = pd.concat([aggregated_df, aggregated_df2,aggregated_df3],ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rankpoints=pd.read_csv('rankpoints_csv.csv', usecols=range(1,7))\n",
    "risk=pd.read_csv('risk_csv.csv')\n",
    "copy_df2=pd.merge(df_123,rankpoints,on=['matchId','teamId'])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
