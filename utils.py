import streamlit as st
import pandas as pd
import numpy as np

def load_status(scenario:str):    
    if (scenario not in st.session_state['scenarios_simulated']):
        # if ('scenario_default' not in st.session_state['scenarios_simulated']):
        df_resource = pd.DataFrame({'Number':[1]},index=['Waiter'])
        df_resource.index.name = 'Resource'
        
        df_location = pd.DataFrame({'Number':[10,2]},index=['Regular Table','Big Table'])
        df_location.index.name = 'Location'
        
        df_total_arrival = pd.DataFrame({'Number':[100]},index=['Total Arrival (#Tables)'])


        arrival_profile_prob = np.array([0,0,0,0,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,0,0,0,0])
        arrival_profile_labels = [f'{idx} - {idx + 1}' for idx in range(arrival_profile_prob.shape[0])]
        df_arrival_hr = pd.DataFrame({'% Clients in each hour':arrival_profile_prob},index=arrival_profile_labels)
        df_arrival_hr.index.name = 'Hour interval'

        df_number_people_table_profile = pd.DataFrame({'Probability Client Tables':np.array([.7,.3])}, index=['Regular table','Big table'])
        
        df_prob_order_dessert = pd.DataFrame({'Probability Table order dessert':np.array([.5,.5])}, index=['Regular table','Big table'])

        time_input = [
            {'index':"Displace until table and sit", 'min':1,'mode':1.5,'max':2}, 
            {'index':'Time to read the menu','min':5,'mode':7,'max':15},
            {'index':"Time to get the order - meal - regular table", 'min':2,'mode':4,'max':7}, 
            {'index':"Time to get the order - meal - big table", 'min':5,'mode':7,'max':10}, 
            {'index':"Time to get the order - drink - regular table", 'min':1,'mode':3,'max':5}, 
            {'index':"Time to get the order - drink - big table", 'min':2,'mode':5,'max':7}, 
            {'index':"Time to get the order - dessert - regular table", 'min':1,'mode':3,'max':5}, 
            {'index':"Time to get the order - dessert - big table", 'min':2,'mode':5,'max':7}, 
            {'index':"Walk until the kitchen", 'min':1,'mode':1.5,'max':2}, 
            {'index':'Time to prepare the order - meal - regular table','min':10,'mode':12,'max':15},
            {'index':'Time to prepare the order - meal - big table','min':15,'mode':18,'max':20},
            {'index':'Time to prepare the order - drink - regular table','min':1,'mode':3,'max':5},
            {'index':'Time to prepare the order - drink - big table','min':2,'mode':5,'max':7},
            {'index':'Time to prepare the order - dessert - regular table','min':1,'mode':3,'max':5},
            {'index':'Time to prepare the order - dessert - big table','min':2,'mode':5,'max':7},
            {'index':'Time to eat - meal - regular table','min':15,'mode':25,'max':30},
            {'index':'Time to eat - meal - big table','min':25,'mode':30,'max':60},
            {'index':'Time to eat - dessert - regular table','min':10,'mode':20,'max':25},
            {'index':'Time to eat - dessert - big table','min':20,'mode':25,'max':30},
        ]
        df_time_input = pd.DataFrame(time_input)

        st.session_state['scenarios_simulated'][scenario] = {
            'df_resource':df_resource,
            'df_location':df_location,
            'df_location':df_location,
            'df_total_arrival':df_total_arrival,
            'df_arrival_hr':df_arrival_hr,
            'df_number_people_table_profile':df_number_people_table_profile,
            'df_prob_order_dessert':df_prob_order_dessert,
            'df_time_input':df_time_input,
        }

        # st.session_state[scenario] = st.session_state['scenario_default'].copy()
        # st.session_state['scenarios_simulated'][scenario] = st.session_state['scenarios_simulated'][scenario]