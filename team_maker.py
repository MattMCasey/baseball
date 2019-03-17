import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
from collections import Counter
from datetime import datetime, timedelta
import constants as const
from constants import team_dict, positions
import urls


def get_batter_adjustments():
    """
    Gets average score per team for adjusting pitcher's likely averages

    Returns dataframe, with columns 'Team' and 'MULT'. Team = team name, MULT
    = the adjusttment for the opposing team's batters
    """
    rq = requests.get(urls.batter_data)
    souped = BeautifulSoup(rq.content, 'html.parser')
    table = souped.find('table', {'class' : 'tr-table datatable scrollable'})
    rows = table.findAll('tr')
    header = [x.text for x in rows[0].findAll('th')]
    content = []

    for row in rows[1:]:
        content.append([x.text for x in row.findAll('td')])

    tmdf = pd.DataFrame(content, columns = header)
    tmdf ['Team'] = [team_dict[x] for x in tmdf.Team]
    tmdf[const.year] = pd.to_numeric(tmdf[const.year])
    tmdf['MULT'] = tmdf[const.year].mean() / tmdf[const.year]
    return tmdf[['Team', 'MULT']]


def get_pitcher_adjustments():
    """
    scrapes data from baseball prospectus and returns a dataframe with a
    multiplier for each pitcher based on their WHIP
    """
    rq = requests.get(urls.pitcher_data)
    souped = BeautifulSoup(rq.content, 'html.parser')
    ttdata = souped.find('table', {'id' : 'TTdata'})
    hdrs = [x.text for x in ttdata.find('tr').findAll('td')]
    data = []

    for r in ttdata.findAll('tr')[1:]:
        data.append([x.text for x in r.findAll('td')])

    pf = pd.DataFrame(data, columns = hdrs)
    numheaders = list(pf)[3:]

    for header in numheaders:
        pf[header] = pd.to_numeric(pf[header])

    pf = pf[pf['IP Start'] > 0]
    pf['WHIP'] = (pf['H'] + pf['BB'] + pf['HBP']) / pf['IP']
    pf['MULT'] = pf['WHIP'] / pf['WHIP'].mean()

    return pf[['NAME', 'MULT']]


def get_projected_players():
    """
    Gets projected players from
    https://rotogrinders.com/lineups/mlb?site=draftkings

    Players who are not projected to play will be filetered out later
    """
    players = []

    rq = requests.get(urls.projected_players)

    souped = BeautifulSoup(rq.content, 'html.parser')

    divs = souped.findAll('div', {'class' : 'blk game'})

    for div in divs:

        hm = div.find('div', {'class' : 'blk home-team'})
        aw = div.find('div', {'class' : 'blk away-team'})

        hp = hm.find('div', {'class' : 'pitcher players'}).find('a', {'class' : 'player-popup'})
        ap = aw.find('div', {'class' : 'pitcher players'}).find('a', {'class' : 'player-popup'})

        ht = hm.find('ul')
        if ht == []:
            ht = hm.find('ul', {'class' : 'players'})

        at = aw.find('ul')
        if at == []:
            at = aw.find('ul', {'class' : 'players'})

        ht = ht.findAll('li', {'class' : 'player'})
        ht = [x.find('span', {'class' : 'pname'}).text.strip() for x in ht]

        at = at.findAll('li', {'class' : 'player'})
        at = [x.find('span', {'class' : 'pname'}).text.strip() for x in at]

        players.extend([hp, ap])
        players.extend(at)
        players.extend(ht)

    return players

def get_injured_players(sport):
    """
    sport = str, the targeted sport
    """

    # Dict of pages for injury reports
    urls = {
        'baseball' : 'https://scores.nbcsports.com/mlb/stats.asp?file=inj'
    }
    url = urls[sport]

    # Getting html of injury report page
    rq = requests.get(url)
    souped = BeautifulSoup(rq.content, 'html.parser')
    tds = souped.find_all('td')

    # Creating list of injured players
    inj = []
    for n in range(5, len(tds), 3):

        td = tds[n].getText()
        if ',' in td:

            # Matching source unicode formatting
            td = unidecode(td.split(',')[0])

            inj.append(td)

    return inj

def prep_dataframe():
    """
    Builds a dataframe from DKSalaries.csv and prepares it
    for use by other functions
    """
    print('Extracting csv')
    df = pd.read_csv('DKSalaries.csv')

    df['PPK'] = df.AvgPointsPerGame / df.Salary * 1000
    print('Filtering by projected players')
    act = get_projected_players()
    df = df[df['Name'].isin(act)]

    # Adding opposing team info
    print('Expanding dataframe')
    all_matchups = []
    for x in df['Game Info']:
        if x == 'Postponed':
            all_matchups.append(['Postponed', 'Postponed'])
        else:
            ats = x.split('@')
            t1 = ats[0]
            t2 = ats[1].split(' ')[0]
            all_matchups.append([t1, t2])

    OpposingTeam = []
    for mu, ta in zip(all_matchups, df.TeamAbbrev):
        if mu[0] == ta:
            OpposingTeam.append(mu[1])
        else:
            OpposingTeam.append(mu[0])

    df['OpposingTeam'] = OpposingTeam

    pdf = df[df['Position'] == 'SP'][['Name', 'TeamAbbrev']]
    pdf.columns = ['OppPitcher', 'OpposingTeam']
    df = pd.merge(df, pdf.reset_index(), how='left', left_on='OpposingTeam',
                    right_on="OpposingTeam")

    print("Getting pitcher and team score information")
    badj = get_batter_adjustments()
    padj = get_pitcher_adjustments()

    pfdf = df[df['Position'] == 'SP']
    bfdf = df[df['Position'] != 'SP']

    pfdf = pd.merge(pfdf, badj, how='left', left_on='OpposingTeam', right_on='Team')
    bfdf = pd.merge(bfdf, padj, how='left', left_on='OppPitcher', right_on='NAME')

    ffdf = pd.concat([pfdf, bfdf])
    ffdf['Adjusted'] = ffdf['AvgPointsPerGame'] * ffdf['MULT']


    return ffdf

"""
FUNCTIONS SECTION 2: PICKING TEAMS
"""

def choose_top_players_pos(df, positions):
    """
    df = DataFrame of all available players
    positions = list of strings, positions to be staffed,
                each position should be in the list the number
                of times it is in the roster

    returns a dataframe of the top n players for each position,
    where n is the number of times the position appears in the list
    """
    df = df.set_index(['Roster Position'])

    # Getting counter dict of positions
    pos_count = Counter(positions)

    # Creating a list of mini dfs with top players by position
    baselist = [df.loc[x][:pos_count[x]] for x in pos_count]

    # Making one big dataframe and concatenating
    return pd.concat(baselist)


def choose_top_players_flex(df, num):
    """
    df = DataFrame of all available players
    num = int, number of players to choose

    returns a dataframe of the top n players, where n == num
    """

    # Removing injured players from the dataframe
    inj = get_injured_players('baseball')
    df = df[~df['Name'].isin(inj)]

    # Returning top players
    return df[:num]


def randomized_best_team(df, positions):
    """
    Randomly picks a team until a picked team is better than 100 teams picked
    after it.
    """

    by_pos = df.set_index('Roster Position')
    by_name = df.set_index('Name')
    pos_counter = Counter(positions)

    beaten = 0
    b_val = 0
    f_roster = []
    total = 0
    outcomes = []

    while beaten < const.beaten_thresh:

        roster = []

        for pos in pos_counter:
            roster.extend(np.random.choice(by_pos.loc[pos].Name, pos_counter[pos], replace=False))

        rdf = by_name.loc[roster]
        sal = rdf.Salary.sum()
        val = rdf.AvgPointsPerGame.sum()

        if sal > 50000:
            beaten += 1
            outcomes.append('sal')
        elif val > b_val:
            f_roster = roster
            b_val = val
            total += beaten
            print(total)
            print(beaten)
            beaten = 0
            outcomes.append('win')

        else:
            outcomes.append('loss')
            beaten += 1

    print(Counter(outcomes))
    return by_name.loc[f_roster]



def make_efficient_team(df, positions = None, num = None):
    """
    df = DataFrame of all available players
    positions = list of strings
    num = int

    Returns a dataframe of a team with the best ratio of
    average points to cost.
    """
    df = df.sort_values('PPK', ascending=False)

    if positions != None:
        return choose_top_players_pos(df, positions)
    if num != None:
        return choose_top_players_flex(df, num)


def make_best_team(df, positions = None, num = None):
    """
    df = DataFrame of all available players
    positions = list of strings
    num = int

    Returns a dataframe of a team with the best average points.
    """

    df = df.sort_values('Adjusted', ascending=False)

    if positions != None:
        return choose_top_players_pos(df, positions)
    if num != None:
        return choose_top_players_flex(df, num)

def make_team_w_pos(positions, df = None):
    """
    positions = list of strings, positions to be staffed,
                each position should be in the list the number
                of times it is in the roster

    returns a dataframe of a team formed with the efficient to
    best method. Starts with the most efficient players for
    their price, then adds the best players, regardless of price,
    while keeping the whole team's price under the cap
    """

    if type(df) == None:
        df = prep_dataframe()

    eff = make_efficient_team(df, positions = positions)
    best = make_best_team(df, positions = positions)

    # 'DEFICIT' = The expectation of best - efficient player
    eff['DEFICIT'] = best.AvgPointsPerGame - eff.AvgPointsPerGame
    best['DEFICIT'] = best.AvgPointsPerGame - eff.AvgPointsPerGame

    eff = eff.sort_values('DEFICIT', ascending=False).reset_index()
    best = best.sort_values('DEFICIT', ascending=False).reset_index()

    # Replacing efficient players with best players in deficit order
    for i, e in eff.iterrows():

        ctotal = eff.Salary.sum()
        ctminus = ctotal - e.Salary

        if ctminus + best.Salary.loc[i] <= 50000:
            if best.Name.loc[i] not in eff.Name.unique():
                eff.loc[i] = best.loc[i]

    print('Salary:', eff.Salary.sum())
    print('Avg Points:', eff.AvgPointsPerGame.sum())
    print('Avg w/o cap:', best.AvgPointsPerGame.sum())
    print('Sal w/o cap:', best.Salary.sum())

    return eff

def make_team_flex(num):
    """
    num = int, number of players to include on the final roster

    returns a dataframe of a team formed with the efficient to
    best method. Starts with the most efficient players for
    their price, then adds the best players, regardless of price,
    while keeping the whole team's price under the cap
    """

    df = prep_dataframe()

    # num quadrupled for 'best' to give more options
    eff = make_efficient_team(df, num =num).reset_index()
    best = make_best_team(df, num = num*4).reset_index()

    # 'DEFICIT' = The expectation of best - efficient player
    eff['DEFICIT'] = best.AvgPointsPerGame - eff.AvgPointsPerGame
    best['DEFICIT'] = best.AvgPointsPerGame - eff.AvgPointsPerGame

    eff = eff.sort_values('DEFICIT', ascending=False).reset_index()
    best = best.sort_values('AvgPointsPerGame', ascending=False).reset_index()

    # Replacing efficient players one at a time while staying under cap
    for i, e in eff.iterrows():

        for j, r in best.iterrows():

            ctotal = eff.Salary.sum()
            ctminus = ctotal - e.Salary

            if ctminus + best.Salary.loc[j] <= 50000 \
            and best.Name.loc[j] not in eff.Name.unique()\
            and e.AvgPointsPerGame < r.AvgPointsPerGame:

                eff.loc[i] = best.loc[j]
                break

    print('Salary:', eff.Salary.sum())
    print('Avg Points:', eff.AvgPointsPerGame.sum())
    print('Avg w/o cap:', best.AvgPointsPerGame[:num].sum())
    print('Sal w/o cap:', best.Salary[:num].sum())

    return eff, best

def clear_dead_weight(df):

    df = df.sort_values('PPK', ascending=False)
    output = pd.DataFrame()

    for pos in [x for x in df.index.unique() if len(x) < 3]:
        print(pos)
        tdf = df.loc[pos]

        floor = tdf.AvgPointsPerGame.iloc[0]
        tdf = tdf[tdf['AvgPointsPerGame'] >= floor]

        output = pd.concat([output, tdf])

    return output

def create_active_player_filter():
    """
    Gets lineups from news sites and returns a list of players who are
    playing tonight.
    """
    rq = requests.get('https://www.mlb.com/starting-lineups').content
    souped = BeautifulSoup(rq, 'html.parser')

    divs = souped.findAll("div", {"class": "starting-lineups__matchup"})

    avails = pd.DataFrame()
    all_sps = []

    for div in divs:
        htm = div.find("span", {"class": "starting-lineups__team-name starting-lineups__team-name--home"}).text.strip()
        atm = div.find("span", {"class": "starting-lineups__team-name starting-lineups__team-name--away"}).text.strip()

        htmlu = div.find("ol", {"class": "starting-lineups__team starting-lineups__team--home"})
        atmlu = div.find("ol", {"class": "starting-lineups__team starting-lineups__team--away"})

        htmbs = htmlu.findAll("li", {"class": "starting-lineups__player"})
        atmbs = atmlu.findAll("li", {"class": "starting-lineups__player"})

        sps = div.findAll("div", {"class" : "starting-lineups__pitcher-name"})
        all_sps.extend([x.text.strip() for x in sps])

        htmbs = pd.DataFrame([x.text.split(' ') for x in htmbs])
        htmbs[4] = htm
        atmbs = pd.DataFrame([x.text.split(' ') for x in atmbs])
        atmbs[4] = atm

        avails = pd.concat([avails, htmbs, atmbs])

    avails.columns=['fname', 'lname', 'b', 'pos', 'team']

    avails['name'] = avails.fname + ' ' + avails.lname

    avails = avails[['name', 'b', 'pos', 'team']]

    anames = []

    for i, x in avails.iterrows():
        if x['b'] == 'Jr.':
            anames.append(x['name'] + ' ' +  x['b'])
        else:
            anames.append(x['name'])

    return all_sps + anames

def make_team_w_pos_ratchet(positions, method = 'points', df = None, column = 'Adjusted'):
    """
    positions = list of strings, positions to be staffed,
                each position should be in the list the number
                of times it is in the roster

    returns a dataframe of a team formed with the efficient to
    best method. Starts with the most efficient players for
    their price, then adds the best players, regardless of price,
    while keeping the whole team's price under the cap
    """

    if type(df) == 'NoneType':
        df = prep_dataframe()

    df = df.sort_values(column, ascending=False)

    best = make_best_team(df, positions = positions)

    swaps = 0

    while best.Salary.sum() > 50000:

        over = best.Salary.sum() - 50000

        swaps += 1
        if swaps % 5 ==0:
            print('swaps:', swaps)
        rmdf = df[~df['Name'].isin(best.Name)]

        loss = []
        alts = []
        diffs = []
        for i, r in best.iterrows():

            tdf = rmdf[rmdf['Position'] == r['Position']]
            tdf = tdf[tdf['Salary'] < r['Salary']]

            if len(tdf) > 0:
                ns, na = tdf.iloc[0][['Salary', column]]
                os, oa = r[['Salary', column]]
                alts.append(tdf.iloc[0])

                if method == 'points':
                    ls = oa - na

                if method == 'ratio':
                    ls = (oa - na) / (os - ns)

                diffs.append([os - ns])
                loss.append(ls)
            else:
                loss.append(10000.5)
                alts.append(None)
                diffs.append([os])

        rep = np.array(loss).argmin()
        best.iloc[rep] = alts[rep]

    print('swaps:', swaps)
    print('Avg points (adj):', best.Adjusted.sum())
    print('Avg points:', best.AvgPointsPerGame.sum())
    print('Salary:', best.Salary.sum())

    return best

def make_n_teams(n, pos, method = 'ratchet'):

    df = prep_dataframe()
    taken = []
    teams = []

    for i in range(n):
        df = df[~df['Name'].isin(taken)]

        team1 = make_team_w_pos_ratchet(pos, df = df)
        team2 = make_team_w_pos(pos, df)

        if team1.Adjusted.sum() > team2.Adjusted.sum():
            print('ratchet')
            teams.append(team1)
        else:
            print('efficient')
            teams.append(team2)
        taken.extend(teams[-1].Name.tolist())

    return teams


def export_team(dfs):
    """
    Makes team that can be uploaded in csv format
    """
    pos = ['P', 'P','C','1B','2B', '3B', 'SS', 'OF', 'OF', 'OF']

    if type(dfs) != 'list':
        df = [dfs]

    ndf = pd.DataFrame()

    for df in dfs:
        df = df.reset_index()

        ids = []
        last = None
        for p in pos:
            if p == last:
                pass
            else:
                last = p
                tdf = df[df['Roster Position'] == p]
                ids.extend(tdf.ID.tolist())

        ndf = pd.concat([ndf, pd.DataFrame([ids])])

    ndf.columns = pos
    ndf.to_csv('current_teams.csv', index=False)

    return ndf
