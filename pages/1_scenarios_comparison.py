import streamlit as st

import pandas as pd
import numpy as np

from typing import Union, Tuple

import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from utils import load_status

def create_cumflow(df:pd.DataFrame,concept_name_plus:str, concept_name_minus:str) -> pd.DataFrame:
    df_table = df.query(f"`concept:name` == '{concept_name_plus}' | `concept:name` == '{concept_name_minus}'")
    df_table['flow_table'] = df_table.apply(lambda x:1 if x['concept:name']==concept_name_plus else -1, axis=1)
    df_table['cumflow_table']=df_table['flow_table'].cumsum()
    return df_table

def create_time_table(df_table:pd.DataFrame, concept_name_plus:str, concept_name_minus:str) -> pd.DataFrame:
    df_time_table = df_table.sort_values(by=['case:concept:name', 'time:timestamp:delta']).pivot(index=['case:concept:name'],columns=['concept:name'],values=['time:timestamp:delta'])
    df_time_table.columns = df_time_table.columns.droplevel()
    df_time_table = df_time_table.reset_index()
    df_time_table['table_captured_h'] = df_time_table[concept_name_plus].apply(lambda x:datetime(2024,2,1) + x)
    df_time_table['time_table'] = df_time_table.apply(lambda x:x[concept_name_minus] - x[concept_name_plus],axis=1)
    df_time_table['time_table'] = df_time_table['time_table'].apply(lambda x:x.total_seconds()/60)

    df_time_table = df_time_table.sort_values(by=[concept_name_plus])    
    return df_time_table

def create_graphs_new(
        df_list:list[pd.DataFrame],
        concept_name_plus:str,
        concept_name_minus:str,
        title_line_cumflow:str="Graph of number of tables occupied along the simulation",
        y_label_line_cumflow:str="Number of tables occupied",
        x_label_line_cumflow:str="Simulation time (h)",
        title_histogram_cumflow:str='Histogram of the number of tables occupied along de simulation',
        xlabel_histogram_cumflow:str='Number of tables occupied',
        title_histogram:str="Histogram of the time duration spent in holding a table (min)",
        xlabel_histogram:str="Time spent holding a table (min)",
        title_line_time_duration:str="Graph of number of tables occupied along the simulation",
        y_label_line_time_duration:str="Number of tables occupied",
        x_label_line_time_duration:str="Simulation time (h)",
        color:str='scenario'
        ) -> pd.DataFrame:
    
    df_table_list = []
    scenario_name_list = []
    for df in df_list:
        df_table_aux = create_cumflow(
            df=df, concept_name_plus=concept_name_plus, concept_name_minus=concept_name_minus
        )  
        df_table_aux['table_captured_h'] = df_table_aux['time:timestamp:delta'].apply(lambda x:datetime(2024,2,1) + x)
        df_table_list.append(df_table_aux)
        scenario_name_list.append(df['scenario'].iloc[0])

    df_table = pd.concat(df_table_list)

    fig_1 = px.line(df_table, x='table_captured_h',y='cumflow_table', color=color, labels={
            "cumflow_table": y_label_line_cumflow,
            "table_captured_h": x_label_line_cumflow,
        },
        title=title_line_cumflow)

    fig_2 = px.histogram(df_table, x="cumflow_table", color=color, labels={
                        "cumflow_table": xlabel_histogram_cumflow,
                    },
                    title=title_histogram_cumflow)
    fig_2.update_layout(
        yaxis_title="# Occurrencies"
    )

    df_time_table_list = []
    for df_table,scenario_name in zip(df_table_list,scenario_name_list):
        df_time_table = create_time_table(df_table = df_table, concept_name_plus = concept_name_plus, concept_name_minus = concept_name_minus)
        df_time_table['scenario'] = scenario_name
        df_time_table_list.append(df_time_table)

    # df_time_table_1 = create_time_table(df_table = df_table_1, concept_name_plus = concept_name_plus, concept_name_minus = concept_name_minus)
    # df_time_table_2 = create_time_table(df_table = df_table_2, concept_name_plus = concept_name_plus, concept_name_minus = concept_name_minus)    
    # df_time_table_1['scenario'] = 'scenario_1'
    # df_time_table_2['scenario'] = 'scenario_2'

    df_time_table = pd.concat(df_time_table_list)

    fig_3 = px.line(df_time_table, x='table_captured_h',y='time_table', color=color, labels={
            "time_table": y_label_line_time_duration,
            "table_captured_h":x_label_line_time_duration
        },
        title=title_line_time_duration)
    # fig.show()

    fig_4 = px.histogram(df_time_table, x="time_table", color=color, labels={
                        "time_table": xlabel_histogram,
                    },
                    title=title_histogram)
    fig_4.update_layout(
        yaxis_title="# Occurrencies"
    )

    # fig.show()

    return fig_1, fig_2, fig_3, fig_4

def create_graphs(
        df_1:pd.DataFrame,
        df_2:pd.DataFrame,
        concept_name_plus:str,
        concept_name_minus:str,
        title_line_cumflow:str="Graph of number of tables occupied along the simulation",
        y_label_line_cumflow:str="Number of tables occupied",
        x_label_line_cumflow:str="Simulation time (h)",
        title_histogram_cumflow:str='Histogram of the number of tables occupied along de simulation',
        xlabel_histogram_cumflow:str='Number of tables occupied',
        title_histogram:str="Histogram of the time duration spent in holding a table (min)",
        xlabel_histogram:str="Time spent holding a table (min)",
        title_line_time_duration:str="Graph of number of tables occupied along the simulation",
        y_label_line_time_duration:str="Number of tables occupied",
        x_label_line_time_duration:str="Simulation time (h)",
        color:str='scenario'
        ) -> pd.DataFrame:
    df_table_1 = create_cumflow(
        df=df_1, concept_name_plus=concept_name_plus, concept_name_minus=concept_name_minus
    )  
    df_table_1['table_captured_h'] = df_table_1['time:timestamp:delta'].apply(lambda x:datetime(2024,2,1) + x)
    df_table_2 = create_cumflow(
        df=df_2, concept_name_plus=concept_name_plus, concept_name_minus=concept_name_minus
    )
    df_table_2['table_captured_h'] = df_table_2['time:timestamp:delta'].apply(lambda x:datetime(2024,2,1) + x)

    df_table = pd.concat([df_table_1,df_table_2])

    fig_1 = px.line(df_table, x='table_captured_h',y='cumflow_table', color=color, labels={
            "cumflow_table": y_label_line_cumflow,
            "table_captured_h": x_label_line_cumflow,
        },
        title=title_line_cumflow)

    fig_2 = px.histogram(df_table, x="cumflow_table", color=color, labels={
                        "cumflow_table": xlabel_histogram_cumflow,
                    },
                    title=title_histogram_cumflow)
    fig_2.update_layout(
        yaxis_title="# Occurrencies"
    )
    # fig.show()

    # df_time_table = df_table.sort_values(by=['scenario', 'case:concept:name', 'time:timestamp:delta']).pivot(index=['case:concept:name'],columns=['concept:name'],values=['time:timestamp:delta'])
    # df_time_table.columns = df_time_table.columns.droplevel()
    # df_time_table = df_time_table.reset_index()
    # df_time_table['table_captured_h'] = df_time_table[concept_name_plus].apply(lambda x:datetime(2024,2,1) + x)
    # df_time_table['time_table'] = df_time_table.apply(lambda x:x[concept_name_minus] - x[concept_name_plus],axis=1)
    # df_time_table['time_table'] = df_time_table['time_table'].apply(lambda x:x.total_seconds()/60)

    # df_time_table = df_time_table.sort_values(by=[concept_name_plus])
    df_time_table_1 = create_time_table(df_table = df_table_1, concept_name_plus = concept_name_plus, concept_name_minus = concept_name_minus)
    df_time_table_2 = create_time_table(df_table = df_table_2, concept_name_plus = concept_name_plus, concept_name_minus = concept_name_minus)    
    df_time_table_1['scenario'] = 'scenario_1'
    df_time_table_2['scenario'] = 'scenario_2'

    df_time_table = pd.concat([df_time_table_1,df_time_table_2])

    fig_3 = px.line(df_time_table, x='table_captured_h',y='time_table', color=color, labels={
            "time_table": y_label_line_time_duration,
            "table_captured_h":x_label_line_time_duration
        },
        title=title_line_time_duration)
    # fig.show()

    fig_4 = px.histogram(df_time_table, x="time_table", color=color, labels={
                        "time_table": xlabel_histogram,
                    },
                    title=title_histogram)
    fig_4.update_layout(
        yaxis_title="# Occurrencies"
    )

    # fig.show()

    return fig_1, fig_2, fig_3, fig_4

if (len(st.session_state['scenarios_simulated'].keys()) > 1):

    df_list = []
    # st.session_state['scenarios_simulated'][scenario_name]
    for scenario_name in st.session_state['scenarios_simulated'].keys():
        try:
            log = st.session_state['scenarios_simulated'][scenario_name]['log']
            df = pd.DataFrame(log)
            df['time:timestamp'] = df['time:timestamp:number'].apply(lambda x:pd.to_datetime(x,unit='m'))
            df['time:timestamp:delta'] = df['time:timestamp:number'].apply(lambda x:pd.to_timedelta(x,unit='m'))
            df['scenario']=scenario_name
            df_list.append(df)

        except:
            st.write(f'{scenario_name} does not have log. Run this scenario to have an output')
        

    tab1, tab2 = st.tabs(["Table Ocuppacy", "Queuing to get a table"])
    with tab1:
        fig_1, fig_2, fig_3, fig_4 = create_graphs_new(df_list=df_list, concept_name_plus='table captured', concept_name_minus='table released')
        # fig_1, fig_2, fig_3, fig_4 = create_graphs(df_1=df_1, df_2=df_2, concept_name_plus='table captured', concept_name_minus='table released')
        st.header("Table Ocuppacy")
        with st.container(height=500):
            st.write(fig_1)
            st.write(fig_2)
            st.write(fig_3)
            st.write(fig_4)
    with tab2:
        # fig_1, fig_2, fig_3, fig_4 = create_graphs(df_1=df_1, df_2=df_2, concept_name_plus='arrived at the model', concept_name_minus='table captured',
        fig_1, fig_2, fig_3, fig_4 = create_graphs_new(df_list=df_list, concept_name_plus='arrived at the model', concept_name_minus='table captured',
            title_line_cumflow="Graph of the line of tables queueing along the simulation",
            y_label_line_cumflow="Number of tables queueing",
            x_label_line_cumflow="Simulation time (h)",
            title_histogram_cumflow='Histogram of the number of tables queueing',
            xlabel_histogram_cumflow='Number of tables queueing',
            title_histogram="Histogram of the time duration spent queueing (min)",
            xlabel_histogram="Time spent holding a table (min)",
            title_line_time_duration="Graph of number of tables occupied along the simulation",
            y_label_line_time_duration="Number of tables occupied",
            x_label_line_time_duration="Simulation time (h)")
        st.header("Queuing to get a table")
        with st.container(height=500):
            st.write(fig_1)
            st.write(fig_2)
            st.write(fig_3)
            st.write(fig_4)
