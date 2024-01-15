import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import numpy as np
pd.set_option('display.max_columns', None)
import plotly.express as px
import pulp as plp
import plotly.subplots as sp

def check_file(file):
	
    xl = pd.ExcelFile(file)
    sheetnames = xl.sheet_names  # see all sheet names

    # Check if the DataFrame has the required column name
    if sorted(sheetnames) != ['laadlocaties', 'laden', 'parameters', 'ritten']:
        error_message = 'het inputbestand moet sheets bevatten met de namen "ritten", "laden", "parameters" en "laadlocaties". Gebruik het template als voorbeeld.'
        st.error(error_message)
        st.stop()
		
    elif list(pd.read_excel(file, sheet_name = 'laden').columns) != ['Activiteit']:
        error_message = 'De sheet "laden" moet alleen een kolom "Activiteit" bevatten met de activiteiten waarvoor laden is toegestaan (zie nieuw template).'
        st.error(error_message)
        st.stop()

def check_file_price(file):
	
    xl = pd.ExcelFile(file)
    sheetnames = xl.sheet_names  # see all sheet names

    # Check if the DataFrame has the required column name
    if sorted(sheetnames) != ['Sheet1']:
        error_message = 'het inputbestand van prijzendata moet de sheet Sheet1 bevatten'
        st.error(error_message)
        st.stop()		

def get_params_template(params_file,accu):
    df_parameters = pd.read_excel(params_file, sheet_name = 'parameters').set_index('naam')
    
    laadvermogen_bedrijf = df_parameters.loc['laadvermogen bedrijf'].waarde
    aansluittijd = df_parameters.loc['aansluittijd'].waarde
    efficiency = df_parameters.loc['efficiency'].waarde
    laadvermogen_snelweg = accu *60
     
    return laadvermogen_bedrijf, aansluittijd, efficiency, laadvermogen_snelweg

def make_dataframes(ritten,prijzen):
    df_ritten = pd.read_excel(ritten)
    df_prijzen = pd.read_excel(prijzen, skiprows=6)
    df_prijzen = df_prijzen.rename(columns={"Unnamed: 0": "Tijden", "Unnamed: 2": "Currency"})
    df_prijzen['Tijdblok'] = df_prijzen['Tijden'].astype(str).str[0] +  df_prijzen['Tijden'].astype(str).str[1]
    return df_ritten, df_prijzen

def krijg_interval_range(row):
          return range(row['Begin_tijd_activiteit_int']+1, row['Eind_tijd_activiteit_int'] + 1)
        
def get_laadlocaties(ritten):
    df_laadlocaties = pd.read_excel(ritten, sheet_name = 'laadlocaties')
    df_laden = pd.read_excel(ritten, sheet_name = 'laden')
    laadlocaties = df_laadlocaties["Positie"].values.tolist()
    laadactiviteiten = df_laden["Activiteit"].values.tolist()
    return laadlocaties, laadactiviteiten

def transform_data(df_ritten,df_prijzen,snelladen,aansluittijd,laadlocaties,laadactiviteiten):
  #Activiteiten uitbreiden zodat deze vallen in tijdseenheden van een uur.
    df_ritten = df_ritten.sort_values(by=['Voertuig','Begindatum en -tijd'])
    #Activiteiten uitbreiden zodat deze vallen in tijdseenheden van een uur.
    df_ritten['Einddatum en -tijd'] = pd.to_datetime(df_ritten['Einddatum en -tijd'])
    df_ritten['Begindatum en -tijd'] = pd.to_datetime(df_ritten['Begindatum en -tijd'])
    df_ritten['Timedelta_begin'] = df_ritten['Begindatum en -tijd'] - df_ritten['Begindatum en -tijd'].min()
    df_ritten['Timedelta_eind'] = df_ritten['Einddatum en -tijd'] - df_ritten['Begindatum en -tijd'].min()
    df_ritten['Begin_tijd_activiteit_int'] = (df_ritten['Timedelta_begin'].dt.total_seconds()/60).astype(int)
    df_ritten['Eind_tijd_activiteit_int'] = (df_ritten['Timedelta_eind'].dt.total_seconds()/60).astype(int)
    df_ritten['Duurmin_activiteit_int'] = df_ritten['Eind_tijd_activiteit_int'] - df_ritten['Begin_tijd_activiteit_int']
    df_ritten['gem_km_per_uur'] = 0
    df_ritten.loc[df_ritten['Activiteit']=='Rijden','gem_km_per_uur'] = df_ritten.loc[df_ritten['Activiteit']=='Rijden', 'Afstand']/(df_ritten.loc[df_ritten['Activiteit']=='Rijden', 'Duurmin_activiteit_int']/60)
    df_ritten = df_ritten.reset_index().rename(columns = {'index' : 'id'})
    
    #Infrastructuur niet rijden
    #Maak lijst aan van posities en activiteiten waar er geladen kan worden
    laadlocaties = laadlocaties
    laadactiviteiten = laadactiviteiten
    df_ritten['Laadpaal_niet_rijden'] = 0
    df_ritten.loc[(df_ritten['Activiteit']!='Rijden')&(df_ritten['Duurmin_activiteit_int']>=aansluittijd/60 + 1)
    &(df_ritten['Positie'].isin(laadlocaties))&(df_ritten['Activiteit'].isin(laadactiviteiten)),'Laadpaal_niet_rijden'] = 1
    df_ritten['Begin_tijd_activiteit_int_met_aansluittijd'] = df_ritten['Begin_tijd_activiteit_int']
    
    df_ritten.loc[df_ritten['Laadpaal_niet_rijden']==1,'Begin_tijd_activiteit_int_met_aansluittijd'] += int(aansluittijd/60)
   
    df_ritten['Begintijd_met_aansluittijd'] = df_ritten['Begindatum en -tijd']
    df_ritten['Begintijd_met_aansluittijd']  = pd.to_datetime(df_ritten['Begintijd_met_aansluittijd']).astype(np.int64)
    df_ritten.loc[df_ritten['Laadpaal_niet_rijden']==1,'Begintijd_met_aansluittijd'] += 600000000000
    df_ritten['Begintijd_met_aansluittijd'] = np.array(df_ritten['Begintijd_met_aansluittijd'], dtype='datetime64[ns]')

    df_ritten['time_range'] = df_ritten.apply(lambda row: list(pd.date_range(row['Begintijd_met_aansluittijd'].ceil('1h'), row['Einddatum en -tijd'].ceil('1h'), freq = '1h')), axis  = 1)
    pd.date_range(df_ritten['Begindatum en -tijd'][0].ceil('1h'), df_ritten['Einddatum en -tijd'][0].ceil('1h'), freq = '1h')
  
    df_ritten_explode = df_ritten.explode(column = 'time_range')
    df_ritten_explode['time_range_shift'] = df_ritten_explode.groupby('id')['time_range'].shift(1)
    df_ritten_explode['begin_output'] = np.where(df_ritten_explode['time_range_shift'].notna(), df_ritten_explode['time_range_shift'], df_ritten_explode['Begintijd_met_aansluittijd'])
    df_ritten_explode['eind_output'] = np.where(df_ritten_explode['time_range'] < df_ritten_explode['Einddatum en -tijd'], df_ritten_explode['time_range'], df_ritten_explode['Einddatum en -tijd'])
    df_ritten_explode['Timedelta_begin'] = df_ritten_explode['begin_output'] - df_ritten_explode['begin_output'].min()
    df_ritten_explode['Timedelta_eind'] = df_ritten_explode['eind_output'] - df_ritten_explode['begin_output'].min()
    df_ritten_explode['Begin_output_int'] = (df_ritten_explode['Timedelta_begin'].dt.total_seconds()/60).astype(int)
    df_ritten_explode['Eind_output_int'] = (df_ritten_explode['Timedelta_eind'].dt.total_seconds()/60).astype(int)
    
    df_ritten_explode['Begintijd'] = df_ritten_explode['begin_output'].dt.time
    df_ritten_explode['Eindtijd'] = df_ritten_explode['eind_output'].dt.time
    df_ritten_explode['Duurmin'] = df_ritten_explode['Eind_output_int'] - df_ritten_explode['Begin_output_int']
    
  
    df_ritten_explode['Tijdblok'] = df_ritten_explode['Begintijd'].astype(str).str[0] + df_ritten_explode['Begintijd'].astype(str).str[1]
    df_ritten_explode['dag'] = df_ritten_explode['begin_output'].astype(str).str[8] + df_ritten_explode['begin_output'].astype(str).str[9]
    df_ritten_explode['dag']=df_ritten_explode['dag'].astype(int)
  
    rows_to_delete = []
    for i in range(24, len(df_prijzen), 27):
        rows_to_delete.extend(range(i, i + 3))

    #Verwijder de geselecteerde rijen
    df_prijzen = df_prijzen.drop(rows_to_delete)

    dagen_per_groep = 24
    aantal_rijen = len(df_prijzen)
    dagen = []

    for rij_nummer in range(aantal_rijen):
        dag_nummer = (rij_nummer// dagen_per_groep)+1
        dagen.append( dag_nummer)
    df_prijzen["dag"] = dagen

    #excelbestanden mergen, energieprijzen aan dataframe toevoegen.
    df_merged = pd.merge(df_ritten_explode,df_prijzen,how = 'left', on = ['Tijdblok','dag'])
    
    #vaste prijs voor snelladen
    df_merged.loc[df_merged['Activiteit'] == 'Rijden', "[EUR/MWh]" ]= snelladen
    return df_ritten, df_merged


def def_variabel(df_ritten,df_merged,afstand):
    laadpaal = 0
    afstand_laadpaal = afstand
    df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Duurmin_activiteit_int']>0),'Afstand_per_rij'] = df_merged.loc[df_merged['Activiteit']=='Rijden','Afstand']*(df_merged.loc[df_merged['Activiteit']=='Rijden','Duurmin'])/df_merged.loc[df_merged['Activiteit']=='Rijden','Duurmin_activiteit_int']
    df_merged.loc[df_merged['Activiteit']=='Rijden','cumulatieve_afstand'] = df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Afstand_per_rij'].notna()),'Afstand_per_rij'].cumsum()
    df_merged.loc[df_merged['Activiteit']=='Rijden','restafstand'] = df_merged.loc[df_merged['Activiteit']=='Rijden','cumulatieve_afstand']%afstand_laadpaal
    df_merged.loc[df_merged['Activiteit']=='Rijden','restafstand'] = df_merged[df_merged['Activiteit']=='Rijden']['restafstand'].shift(1)
    index_eerste_rij_rijden = df_merged.loc[df_merged['Activiteit']=='Rijden'].index[0]
    df_merged.at[index_eerste_rij_rijden,'restafstand'] = 0
    df_merged.loc[df_merged['Activiteit']=='Rijden','Aantal_laadpalen'] = ((df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Afstand_per_rij'].notna())&(df_merged['restafstand'].notna()),'Afstand_per_rij'] + df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Afstand_per_rij'].notna())&(df_merged['restafstand'].notna()),'restafstand'])/afstand_laadpaal).astype(int)
    df_merged.loc[df_merged['Activiteit']=='Rijden','Tijdstip_laadpaal'] = 0

    naam_kolommen_laadpalen_rijden = []
    for laadpaal in range(1,int(df_merged.loc[df_merged['Activiteit']=='Rijden','Aantal_laadpalen'].max())+1):
      tijdstip_eerste_laadpaal = df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Aantal_laadpalen']>=1),'Begin_output_int'] + (afstand_laadpaal - df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Aantal_laadpalen']>=1),'restafstand'])/df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Aantal_laadpalen']),'gem_km_per_uur']*60
      if laadpaal == 1:
        df_merged['Tijdstip_laadpaal_1'] = 0
        df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Aantal_laadpalen']>=1),'Tijdstip_laadpaal_1'] = (tijdstip_eerste_laadpaal+0.5).astype(int)
        naam_kolommen_laadpalen_rijden.append('Tijdstip_laadpaal_1')
        continue
      else:
        naam_kolom = 'Tijdstip_laadpaal_' + str(laadpaal)
        naam_kolommen_laadpalen_rijden.append(naam_kolom)
        df_merged[naam_kolom] = 0
        tijdstip_laadpaal = tijdstip_eerste_laadpaal + (laadpaal-1)*afstand_laadpaal/df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Aantal_laadpalen']>=laadpaal),'gem_km_per_uur']*60
      df_merged.loc[(df_merged['Activiteit']=='Rijden')&(df_merged['Aantal_laadpalen']>=laadpaal),naam_kolom] = tijdstip_laadpaal
    df_merged[naam_kolom] = (df_merged[naam_kolom]+0.5).astype(int)

    df_merged = df_merged[(df_merged['Laadpaal_niet_rijden']==1)|((df_merged['Activiteit']=='Rijden')&(df_merged['Afstand_per_rij']>0))]
    #Maak dataset aan met tijdstippen waar een laadpaal is voor tijdens het rijden
    lijst_tijstip_laadpalen_rijden = []
    for i in range(1,laadpaal+1):
      tijstip_laadpalen_rijden = df_merged.loc[df_merged['Tijdstip_laadpaal_'+str(i)]>0,['Voertuig','Tijdstip_laadpaal_'+str(i)]]
      tijstip_laadpalen_rijden.rename(columns={'Tijdstip_laadpaal_'+str(i): 'Tijdstip_laadpaal'}, inplace = True)
      lijst_tijstip_laadpalen_rijden.append(tijstip_laadpalen_rijden)
    if laadpaal > 0:
      df_tijdstippen_laadpalen_rijden = pd.concat(lijst_tijstip_laadpalen_rijden)

    #Maak dataset aan met tijdstippen waar een laadpaal is voor tijdens het niet rijden
    df = df_ritten.loc[df_ritten['Laadpaal_niet_rijden']==1,['Voertuig','Begin_tijd_activiteit_int','Eind_tijd_activiteit_int']]
    if len(df) > 0:
      
      df['interval_range'] = df.apply(krijg_interval_range, axis=1)
      df_expanded = df.explode('interval_range')
      df_expanded.reset_index(drop=True, inplace=True)
      df_tijdstippen_laadpalen_niet_rijden = df_expanded[['Voertuig','interval_range']]
      df_tijdstippen_laadpalen_niet_rijden.rename(columns={'interval_range': 'Tijdstip_laadpaal'},inplace = True)

      #Voeg beide datasets voor de laadpalen samen
      if laadpaal > 0:
        df_I_J = pd.concat([df_tijdstippen_laadpalen_rijden, df_tijdstippen_laadpalen_niet_rijden])
        df_I_J = df_I_J.drop_duplicates()
      else:
        df_I_J = df_tijdstippen_laadpalen_niet_rijden
      df_I_J.sort_values('Tijdstip_laadpaal' ,inplace=True)
      df_I_J = df_I_J.reset_index()
    else:
      df_I_J = df_tijdstippen_laadpalen_rijden
      df_I_J = df_I_J.reset_index()
    return df_merged, naam_kolommen_laadpalen_rijden, df_I_J
  
@st.cache_data
def make_model(df_merged,naam_kolommen_laadpalen_rijden,df_I_J,accu,efficiency,laadvermogen_snelweg,laadvermogen_bedrijf):
    #Maak lijsten van de eerste en laatste rijen van alle voertuigen
    eerste_rij_van_elk_voertuig = []
    laatste_rij_van_elk_voertuig = []
    relevante_df_merged = relevante_df_merged = df_merged[(df_merged['Laadpaal_niet_rijden']==1)|((df_merged['Activiteit']=='Rijden')&((df_merged['Afstand_per_rij']>0)))]

    for voertuig in df_merged['Voertuig'].unique():
      eerste_rij_van_elk_voertuig.append(df_merged.loc[df_merged['Voertuig']==voertuig].iloc[0].name)

    for voertuig in relevante_df_merged['Voertuig'].unique():
      laatste_rij_van_elk_voertuig.append(relevante_df_merged.loc[relevante_df_merged['Voertuig']==voertuig].iloc[-1].name)
    
    
    #x_variabelen
    opt_model = plp.LpProblem(name='MIP_Model')

    x_vars  = {(df_I_J['Voertuig'][row],df_I_J['Tijdstip_laadpaal'][row]):
        plp.LpVariable(cat=plp.LpContinuous,lowBound=0, upBound=1, name="x_{0}_{1}".format(df_I_J['Voertuig'][row],df_I_J['Tijdstip_laadpaal'][row])) 
        for row in range(len(df_I_J))}

    
    voertuig = df_merged.iloc[0]['Voertuig']
    
    #Bepaal volheid accu van voertuigen gedurende ritten
    #Zorg ervoor dat elk voertuig start met een volle accu
    df_merged['Accu'] = 0
    df_merged['Laadtijd_min'] = 0
    for voertuig in df_merged['Voertuig'].unique():
        for i in df_merged[df_merged["Voertuig"]==voertuig].index:
            if i in eerste_rij_van_elk_voertuig:
                df_merged.at[i,'Accu'] = accu
                break
    #Bepaal hoeveel kWh gebruikt wordt tijdens het rijden en trek het van de accu af
    df_merged.loc[df_merged['Activiteit']=='Rijden','Accu'] -= df_merged.loc[df_merged['Activiteit']=='Rijden','Afstand_per_rij']*efficiency
    #Voeg mogelijke laadmomenten toe aan de accu voor tijdens het rijden
    df_laadmomenten_rijden = df_merged[df_merged['Aantal_laadpalen']>0]
    df_laadmomenten_rijden['Aantal_laadpalen'] = df_laadmomenten_rijden['Aantal_laadpalen'].astype(int)
    max_aantal_laadpalen_per_rij = df_laadmomenten_rijden.loc[df_laadmomenten_rijden['Activiteit']=='Rijden','Aantal_laadpalen'].max()
    for laadpaal in range(max_aantal_laadpalen_per_rij):
        naam_kolom_laadpaal = naam_kolommen_laadpalen_rijden[laadpaal]
        def update_accu(row):
            voertuig = row['Voertuig']
            laadmoment = row[naam_kolom_laadpaal]
            return plp.lpSum([x_vars[voertuig,laadmoment]*laadvermogen_snelweg/60,row['Accu']])
        def krijg_laadtijd_min(row):
            voertuig = row['Voertuig']
            laadmoment = row[naam_kolom_laadpaal]
            return plp.lpSum([x_vars[voertuig,laadmoment]])
        df_merged.loc[df_merged['Aantal_laadpalen']==laadpaal+1,'Accu'] = df_laadmomenten_rijden.loc[df_laadmomenten_rijden['Aantal_laadpalen']==laadpaal+1].apply(update_accu, axis=1)
        df_merged.loc[df_merged['Aantal_laadpalen']==laadpaal+1,'Laadtijd_min'] = df_laadmomenten_rijden.loc[df_laadmomenten_rijden['Aantal_laadpalen']==laadpaal+1].apply(krijg_laadtijd_min, axis=1)
      #Voeg mogelijke laadmomenten toe aan de accu voor tijdens het stilstaan
    df_laadmomenten_stilstaan = df_merged.loc[df_merged['Laadpaal_niet_rijden']==1,['Voertuig','Begin_output_int','Eind_output_int']]
    def update_accu(row):
        voertuig = row['Voertuig']
        laadmomenten = range(row['Begin_output_int']+1,row['Eind_output_int']+1)
        return plp.lpSum([x_vars[voertuig,laadmoment]*laadvermogen_bedrijf/60 for laadmoment in laadmomenten])
    def krijg_laadtijd_min(row):
        voertuig = row['Voertuig']
        laadmomenten = range(row['Begin_output_int']+1,row['Eind_output_int']+1)
        return plp.lpSum([x_vars[voertuig,laadmoment] for laadmoment in laadmomenten])
    df_merged.loc[df_merged['Laadpaal_niet_rijden']==1,'Accu'] += df_laadmomenten_stilstaan.apply(update_accu, axis=1)
    df_merged.loc[df_merged['Accu'].isna(),'Accu']=0
    df_merged.loc[df_merged['Laadpaal_niet_rijden']==1,'Laadtijd_min'] = df_laadmomenten_stilstaan.apply(krijg_laadtijd_min, axis=1)
    df_merged.loc[df_merged['Laadtijd_min'].isna(),'Laadtijd_min']=0
 
    df_merged['Accu_cum'] = 0
    for voertuig in df_merged['Voertuig'].unique():
        df_merged.loc[df_merged['Voertuig']==voertuig,'Accu_cum'] = df_merged.loc[df_merged['Voertuig']==voertuig,'Accu'].cumsum()
    df_merged['Accu'] = df_merged['Accu_cum'] 
    
    #randvoorwaarden
    for i in df_merged.index:
      if df_merged["Activiteit"][i] =="Rijden":
       #randvoorwaarde 1
        opt_model.addConstraint(
        plp.LpConstraint(e = df_merged["Accu"][i],
                   sense=plp.LpConstraintGE,
                   rhs=0,
                   name="constraint1_{0}".format(i)))
      #randvoorwaarde 2            
      opt_model.addConstraint(
        plp.LpConstraint(e=df_merged["Accu"][i],
                   sense=plp.LpConstraintLE,
                   rhs=accu,
                   name="constraint2_{0}".format(i)))   
      
    for i in laatste_rij_van_elk_voertuig:
      opt_model.addConstraint(
          plp.LpConstraint(e= df_merged["Accu"][i],
                     sense=plp.LpConstraintGE,
                     rhs=accu - 100,
                     name="constraint3_{0}".format(i)))
      
    result = 0
    relevante_df_merged = df_merged[(df_merged['Laadpaal_niet_rijden']==1)|(df_merged['Aantal_laadpalen']>0)]
    max_aantal_laadpalen_per_rij = int(relevante_df_merged.loc[relevante_df_merged['Activiteit']=='Rijden','Aantal_laadpalen'].max())
    #Kosten voor opladen snelweg
    for laadpaal in range(max_aantal_laadpalen_per_rij):
      naam_kolom_laadpaal = naam_kolommen_laadpalen_rijden[laadpaal]
      laadmomenten = relevante_df_merged.loc[relevante_df_merged['Aantal_laadpalen']==laadpaal+1,['Voertuig',naam_kolom_laadpaal]]
      prijs_per_kWh = relevante_df_merged.loc[relevante_df_merged['Aantal_laadpalen']==laadpaal+1,'[EUR/MWh]']/1000
      for row in laadmomenten.index:
        result += plp.lpSum(x_vars[laadmomenten['Voertuig'][row],laadmomenten[naam_kolom_laadpaal][row]]*prijs_per_kWh[row]*laadvermogen_snelweg/60)
    #Kosten voor opladen bij een bedrijf
    laadmomenten = relevante_df_merged.loc[relevante_df_merged['Laadpaal_niet_rijden']==1,['Voertuig','Begin_output_int','Eind_output_int']]
    prijs_per_kWh = relevante_df_merged.loc[relevante_df_merged['Laadpaal_niet_rijden']==1,'[EUR/MWh]']/1000
    for row in laadmomenten.index:
      result += plp.lpSum(x_vars[laadmomenten['Voertuig'][row],laadmoment] for laadmoment in range(laadmomenten['Begin_output_int'][row]+1,laadmomenten['Eind_output_int'][row]+1))*prijs_per_kWh[row]*laadvermogen_bedrijf/60
    
    objective = result
    opt_model.sense = plp.LpMinimize
    opt_model.setObjective(objective)
    check_model = opt_model.solve()
    
    opt_model.solve()

    #Resultaten
    #Voeg kolom Laadtijd_min, kosten, accu en opladen toe
    df_merged['Accu_waarde'] = df_merged.loc[~df_merged['Accu'].apply(lambda x: isinstance(x, (int,float))),'Accu'].apply(lambda expr: expr.value())
    df_merged.loc[df_merged['Accu_waarde'].isna(),'Accu_waarde'] = df_merged.loc[df_merged['Accu_waarde'].isna(),'Accu']
    df_merged['Accu'] = df_merged['Accu_waarde']
    df_merged.drop('Accu_waarde', axis=1, inplace=True)
    df_merged['Laadtijd_min'] = df_merged.loc[~df_merged['Laadtijd_min'].apply(lambda x: isinstance(x, (int,float))),'Laadtijd_min'].apply(lambda expr: expr.value())
    df_merged.loc[df_merged['Laadtijd_min'].isna(),'Laadtijd_min'] = 0
 
    df_merged['Kosten'] = 0
    df_merged.loc[df_merged['Activiteit']=='Rijden','Kosten'] = df_merged.loc[df_merged['Activiteit']=='Rijden','Laadtijd_min']*(df_merged.loc[df_merged['Activiteit']=='Rijden','[EUR/MWh]']/1000)*laadvermogen_snelweg/60
    df_merged.loc[df_merged['Activiteit']!='Rijden','Kosten'] = df_merged.loc[df_merged['Activiteit']!='Rijden','Laadtijd_min']*(df_merged.loc[df_merged['Activiteit']!='Rijden','[EUR/MWh]']/1000)*laadvermogen_bedrijf/60
    
    df_merged['Cumulatieve_kosten'] = df_merged.groupby("Voertuig")["Kosten"].cumsum()
    #resultaten opslaan
    voertuigkosten = pd.DataFrame(columns=["Voertuig","Totale_kosten_euro","Kosten_per_100_gereden_km"])
    for voertuig in df_merged['Voertuig'].unique():
        prijs_per_km = df_merged[df_merged['Voertuig'] == voertuig]['Kosten'].sum() / df_merged[df_merged['Voertuig'] == voertuig]['Afstand'].sum() * 100
        kosten = df_merged[df_merged['Voertuig'] == voertuig]['Kosten'].sum()
        if isinstance(voertuig,str):
          voertuigkosten.loc[len(voertuigkosten)]=[voertuig,round(kosten,2),round(prijs_per_km,2)]
        else:  
          voertuigkosten.loc[len(voertuigkosten)]=[round(voertuig/1),round(kosten,2),round(prijs_per_km,2)]
    
    return df_merged,check_model,voertuigkosten
  
  



def show_plots_accu(df_voertuig):
    fig = px.line(df_voertuig, x='eind_output', y='Accu', title='Accu Waardes Over Tijd')
    fig.update_xaxes(title_text='Tijd')
    st.plotly_chart(fig)
    
def show_plots_kosten(df_voertuig):
    fig = px.line(df_voertuig, x='eind_output', y='Kosten', title='Kosten Over Tijd')
    fig.update_xaxes(title_text='Tijd')
    st.plotly_chart(fig)
    
# Functie voor de analyse figuren
def Analyse(data):
    # Eerste plot - Gemiddelde energieprijzen per tijdsblok
    data['Tijdblok'] = data['Tijdblok'].astype(int)
    gemiddelde_per_tijdsblok = data[data['Activiteit']!='Rijden'].groupby('Tijdblok')['[EUR/MWh]'].mean()
    fig1 = px.bar(x=gemiddelde_per_tijdsblok.index, y=gemiddelde_per_tijdsblok)
 
    # Tweede plot - Aantal oplaadmomenten per tijdsblok
    kosten_histogram = data[(data['Laadtijd_min'] > 0)&(data['Activiteit']!='Rijden')].groupby('Tijdblok')['Laadtijd_min'].sum()
    x = [i for i in range(24)]
    y = [kosten_histogram[i] if i in kosten_histogram.index else 0 for i in range(24)]
    fig2 = px.bar(x=x, y=y)
 
    fig = sp.make_subplots(rows=1, cols=2,horizontal_spacing= 0.1, subplot_titles=('Gemiddelde energieprijs per tijdsblok over de maand', 'Laadminuten per tijdsblok over de maand'))
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)
    fig.update_layout(showlegend=False, xaxis_title='Tijdsblok', yaxis_title='Gemiddelde energieprijs', xaxis2_title='Het tijdsblok waarin opgeladen wordt', yaxis2_title='Aantal')
    st.plotly_chart(fig)

def Analyse_individueel(data,optie):
    # Eerste plot - Gemiddelde energieprijzen per tijdsblok
    data['Tijdblok'] = data['Tijdblok'].astype(int)
    gemiddelde_per_tijdsblok = data[data['Activiteit']!='Rijden'].groupby('Tijdblok')['[EUR/MWh]'].mean()
    fig1 = px.bar(x=gemiddelde_per_tijdsblok.index, y=gemiddelde_per_tijdsblok)
 
    # Tweede plot - Aantal oplaadmomenten per tijdsblok
    kosten_histogram = data[(data['Laadtijd_min'] > 0)&(data['Activiteit']!='Rijden')].groupby('Tijdblok')['Laadtijd_min'].sum()
    x = [i for i in range(24)]
    y = [kosten_histogram[i] if i in kosten_histogram.index else 0 for i in range(24)]
    fig2 = px.bar(x=x, y=y)
 
    fig = sp.make_subplots(rows=1, cols=2,horizontal_spacing= 0.1, subplot_titles=('Gemiddelde energieprijs per tijdsblok over de maand', 'Laadminuten per tijdsblok over de maand'))
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)
    fig.update_layout(showlegend=False, xaxis_title='Tijdsblok', yaxis_title='Gemiddelde energieprijs', xaxis2_title='Het tijdsblok waarin opgeladen wordt', yaxis2_title='Aantal')
    return fig

def download_excel(df):

    st.subheader('Definitieve dataset')
	
    @st.cache_data
    def create_output_excel(df):
        df = df.drop(['Datum'], axis = 1)
        excel_data = BytesIO()
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data.seek(0)
        return excel_data

    # Offer the file download
    st.download_button('Download Excelbestand met modeluitkomsten', create_output_excel(df), file_name='laadmodel_resultaten.xlsx')


def download_template_ritten():
    # Template file URL
    template_ritten_url = 'https://github.com/DatalabHvA/laadmodel/raw/main/template.xlsx'

    # Request the template file
    response = requests.get(template_ritten_url)

    # Create a BytesIO object
    template_ritten_data = BytesIO(response.content)

    # Offer the file download
    st.download_button('Download Template ritten', template_ritten_data, file_name='template_ritten.xlsx')


def download_template_prijzen():
    # Template file URL
    template_prijzen_url = 'https://github.com/yogem99/Bedrijfscase4.2/raw/main/template%20prijzen.xlsx'

    # Request the template file
    response = requests.get(template_prijzen_url)

    # Create a BytesIO object
    template_prijzen_data = BytesIO(response.content)

    # Offer the file download
    st.download_button('Download Template prijzen', template_prijzen_data, file_name='template_prijzen.xlsx')



def main():
    st.title('Laadmodel elektrisch opladen')
    st.write("De resultaten van deze tool zijn informatief.  \nDe verstrekte informatie kan onvolledig of niet geheel juist zijn.  \nAan de resultaten van deze tool kunnen geen rechten worden ontleend.")

    # Download template button
    download_template_ritten()

    # File upload
    uploaded_file_ritten = st.file_uploader('Upload Excelbestand met rittendata, gestructueerd zoals de template voor rittendata', type=['xlsx'])
    
    
    if uploaded_file_ritten is not None:
        laadlocaties,laadactiviteiten = get_laadlocaties(uploaded_file_ritten)
        laadlocaties = st.multiselect('Selecteer op welke laadlocaties er geladen mag worden.',laadlocaties,laadlocaties)
        laadactiviteiten = st.multiselect('Selecteer tijdens welke activiteiten er geladen mag worden.',laadactiviteiten,laadactiviteiten)
        
        download_template_prijzen()
        uploaded_file_prices = st.file_uploader('Upload Excelbestand met prijzendata, gestructueerd zoals de template voor prijzendata (afkomsting van Enstoe).', type=['xlsx'])
    
    if  uploaded_file_ritten and uploaded_file_prices is not None:
        snelladen = st.number_input(label = 'Voer de prijs (euro/MWh) voor snelladen in, tussen de 0 en 600 euro.',min_value= 0,max_value=600,value = 300)
        accu = st.number_input(label = 'Voer de waarde voor accu van vrachtwagen in, tussen 0 en 1500.',min_value= 0,max_value=1500,value = 900)
        afstand = st.slider(label = 'Geef de afstand (km) tussen laadpalen aan, tussen 0 en 100.',min_value= 0,max_value=100,value = 50,step = 10)
        
        try:
            check_file(uploaded_file_ritten)
            check_file_price(uploaded_file_prices)
            laadvermogen_bedrijf, aansluittijd, efficiency, laadvermogen_snelweg = get_params_template(uploaded_file_ritten,accu)
        
        
            df_ritten, df_prijzen = make_dataframes(uploaded_file_ritten,uploaded_file_prices)
        
            st.header('Modelresultaten:')
        
            df_ritten,df_merged = transform_data(df_ritten,df_prijzen,snelladen,aansluittijd,laadlocaties,laadactiviteiten)
            df_merged, naam_kolommen_laadpalen_rijden, df_I_J = def_variabel(df_ritten,df_merged,afstand)
            
            df_merged,check_model,voertuigkosten = make_model(df_merged,naam_kolommen_laadpalen_rijden,df_I_J,accu,efficiency,laadvermogen_snelweg,laadvermogen_bedrijf)
          
            if check_model == -1:
                st.error("Met de huidige inputwaardes kan er geen oplossing gevonden worden, probeer de bijvoorbeeld de accu groter te maken")
                st.stop()
          
            st.dataframe(voertuigkosten)
            st.write("Hieronder 2 tabellen voor gemiddelde energieprijs en laadmomenten van alle vrachtwagen gedurende de gehele maand wanneer de vrachtwagen niet aan het rijden is.")
            Analyse(df_merged)
            st.write("Voor verdieping kan hier elk vrachtwagen verder geanalyseerd worden.")
            opties_voertuigen = st.selectbox(label = "Kies één van de " + str(len(df_merged["Voertuig"].unique())) +  " voertuigen om hierop te verdiepen.", options = df_merged["Voertuig"].unique())
            
            tab1, tab2, tab3 = st.tabs(["🔋 Accu", "💰 Kosten","🚚 Laadmomenten"])
            tab1.subheader('Accu waarden over tijd')
            fig1 = px.line(df_merged[df_merged['Voertuig'] ==opties_voertuigen] , x='eind_output', y='Accu', labels = {'eind_output' : "Datum" })
            tab1.plotly_chart(fig1)
            
            
            tab2.subheader('Kosten over tijd')
            fig2 = px.line(df_merged[df_merged['Voertuig'] ==opties_voertuigen], x='eind_output', y='Cumulatieve_kosten', labels = {'eind_output' : "Datum" })
            tab2.plotly_chart(fig2)
            
            tab3.subheader('Laadmomenten wanneer vrachtwagen niet aan rijden is.')
            fig3 = Analyse_individueel(df_merged[df_merged['Voertuig'] ==opties_voertuigen],opties_voertuigen)
            tab3.plotly_chart(fig3)
         
        except Exception as e:
            st.error(f'Error processing the file: {e}')
    

if __name__ == '__main__':
    main()
