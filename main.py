import streamlit as st

import pandas as pd
import numpy as np
import simpy
from distfit import distfit
from scipy.stats import triang
from simpy.resources.resource import Request, Resource
from typing import Union, Tuple
import pm4py
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
from utils import load_status
import re
import os


st.markdown(
    f'''
        # Restaurant simulation
        ### The power of the combination between discrete event simulation with SimPy and Process Mining with PM4PY
    '''
)

if "scenarios_simulated" not in st.session_state:
    st.session_state["scenarios_simulated"] = {}
if "scenarios_defined" not in st.session_state:
    st.session_state["scenarios_defined"] = []
if "current_scenario_idx" not in st.session_state:
    st.session_state["current_scenario_idx"] = 0
if "last_scenario_loaded" not in st.session_state:
    st.session_state["last_scenario_loaded"] = None

with st.expander("Description"):
    st.markdown(
        '''
            The simulation model represents a typical restaurant experience from the perspective of a customer. The process begins when a customer arrives at the restaurant. The customer’s arrival triggers the start of the simulation.

            Upon arrival, the customer waits for an available table. The tables are categorized into two types: regular and big. The selection of the table depends on the customer’s preference and the availability of the tables. This is modeled using SimPy’s resources to represent the tables.

            Once a table becomes available, the customer sits at the table and calls the waiter. The waiter’s response time can be modeled as a random variable to add realism to the simulation.

            After the waiter arrives, the customer spends some time reading the menu. This duration can also be represented as a random variable. Once the customer has decided, they place their order, which includes a drink, a meal, and possibly a dessert.

            The preparation of the order is the next stage in the simulation. Each item (drink, meal, dessert) has a different preparation time, which can be modeled using SimPy’s delay or timeout functionality.

            Once the order is ready, the waiter serves it to the customer. The customer then spends some time eating and enjoying their meal. This duration can be modeled as another random variable.

            After finishing the meal, the customer asks for the bill, pays it, and then leaves the restaurant. The departure of the customer marks the end of the simulation for that customer.

            This simulation model can be run for many customers to understand the dynamics of the restaurant, identify bottlenecks, and optimize the restaurant’s operations. The power of SimPy lies in its ability to model complex systems and processes in a relatively straightforward manner, making it an excellent tool for this kind of simulation.

            Please note that this is a high-level description. The actual implementation would involve writing Python code using the SimPy library and may require additional details based on the specific requirements of the project.        

        ''')

with st.container(border=True):

    scenario_name_new = st.text_input('Define a new scenario name', placeholder='type the scenario name here')
    btn = st.button('new scenario')
    if btn:
        if scenario_name_new not in st.session_state['scenarios_defined']:
            st.session_state['scenarios_defined'].append(scenario_name_new)
            st.rerun()
        else:
            st.warning('Scenario already existed')

    scenario_name = st.selectbox('Select the scenario you want to run', options=st.session_state["scenarios_defined"], placeholder='type the scenario name here', key='current_scenario', index=st.session_state["current_scenario_idx"])

    try:
        st.session_state["current_scenario_idx"] = st.session_state["scenarios_defined"].index(st.session_state["current_scenario"])
    except:
        pass

    if (not scenario_name):
        scenario_name = 'scenario_default'
    else:
        if st.session_state["last_scenario_loaded"] != scenario_name:
            st.session_state["last_scenario_loaded"] = scenario_name
            st.rerun()

    load_status(scenario_name)


df_resource = st.session_state['scenarios_simulated'][scenario_name]['df_resource']
df_location = st.session_state['scenarios_simulated'][scenario_name]['df_location']
df_location = st.session_state['scenarios_simulated'][scenario_name]['df_location']
df_total_arrival = st.session_state['scenarios_simulated'][scenario_name]['df_total_arrival']
df_arrival_hr = st.session_state['scenarios_simulated'][scenario_name]['df_arrival_hr']
df_number_people_table_profile = st.session_state['scenarios_simulated'][scenario_name]['df_number_people_table_profile']
df_prob_order_dessert = st.session_state['scenarios_simulated'][scenario_name]['df_prob_order_dessert']
df_time_input = st.session_state['scenarios_simulated'][scenario_name]['df_time_input']


def register_log(log:list[dict],case_id:int,activity:str,timestamp:Union[str,int],**kargs):
    instance = {'case:concept:name':str(case_id), 'concept:name':activity, 'time:timestamp:number':timestamp}
    instance.update(kargs)
    log.append(instance)
    return log

def triang_values(min_value,mode_value, max_value) -> np.ndarray:
    return triang.rvs(c=(mode_value-min_value)/(max_value-min_value), loc = min_value, scale=max_value-min_value, size=1)[0]

def get_resource_from_list(env:simpy.Environment, docks:list[Resource]) -> Tuple[Resource,int,Request]:
    dock_reqs = [dock.request() for dock in docks]

    dock_reqs_dict = {dock_req:[dock,idx] for idx,(dock,dock_req) in enumerate(zip(docks,dock_reqs))}

    result = yield env.any_of(dock_reqs)
    # Select one of the triggered requests
    selected = list(result.keys())[0]

    for req in dock_reqs:
        if req != selected and req in result:
            # Release the other triggered requests
            req.resource.release(req)
        elif req not in result:
            # Cancel the remaining requests
            req.cancel()    
    return dock_reqs_dict[selected][0], dock_reqs_dict[selected][1], selected

# draw a numpy array values from a prob array
def draw_values_from_profile(arrival_profile_prob:Union[np.ndarray,None], number=100) -> np.ndarray:
    # arrival_profile_prob = [.1,0,.02, ... , .14]
    # arrival_profile_prob sum 100%
    if (arrival_profile_prob is None):
        arrival_profile_prob=np.random.rand(24)
    arrival_profile_prob = arrival_profile_prob/arrival_profile_prob.sum()
    arrival_profile_prob[-1] = 1 - arrival_profile_prob[:-1].sum()

    return np.array([np.random.choice(range(arrival_profile_prob.shape[0]),p=arrival_profile_prob) for _ in range(number)])

def generate_output_dfg(df:pd.DataFrame,entity:str):
    # frequency
    dfg, start_activities, end_activities = pm4py.discover_dfg(df.query(f"entity == '{entity}'"))
    viz_count = pm4py.visualization.dfg.visualizer.apply(dfg,df.query(f"entity == '{entity}'"), parameters={"start_activities": start_activities, "end_activities": end_activities,"format": "png"})
    viz_count.graph_attr.update({'rankdir': 'TB'})

    # performance
    performance_dfg, start_activities, end_activities = pm4py.discover_performance_dfg(df.query(f"entity == '{entity}'"))
    viz_performance = pm4py.visualization.dfg.visualizer.apply(performance_dfg, variant=pm4py.visualization.dfg.visualizer.Variants.PERFORMANCE, parameters={"start_activities": start_activities, "end_activities": end_activities})
    viz_performance.graph_attr.update({'rankdir': 'TB'})

    return viz_count, viz_performance

def create_cumflow(df:pd.DataFrame,concept_name_plus:str, concept_name_minus:str) -> pd.DataFrame:
    df_table = df.query(f"`concept:name` == '{concept_name_plus}' | `concept:name` == '{concept_name_minus}'")
    df_table['flow_table'] = df_table.apply(lambda x:1 if x['concept:name']==concept_name_plus else -1, axis=1)
    df_table['cumflow_table']=df_table['flow_table'].cumsum()
    return df_table


def create_graphs(
        df:pd.DataFrame,
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
        x_label_line_time_duration:str="Simulation time (h)"
        ) -> pd.DataFrame:
    df_table = create_cumflow(
        df=df, concept_name_plus=concept_name_plus, concept_name_minus=concept_name_minus
    )  

    df_table['table_captured_h'] = df_table['time:timestamp:delta'].apply(lambda x:datetime(2024,2,1) + x)
    fig_1 = px.line(df_table, x='table_captured_h',y='cumflow_table', labels={
            "cumflow_table": y_label_line_cumflow,
            "table_captured_h": x_label_line_cumflow,
        },
        title=title_line_cumflow)
    # fig_1.show()

    fig_2 = px.histogram(df_table, x="cumflow_table",labels={
                        "cumflow_table": xlabel_histogram_cumflow,
                    },
                    title=title_histogram_cumflow)
    fig_2.update_layout(
        yaxis_title="# Occurrencies"
    )
    # fig.show()

    df_time_table = df_table.sort_values(by=['case:concept:name','time:timestamp:delta']).pivot(index=['case:concept:name'],columns=['concept:name'],values=['time:timestamp:delta'])
    df_time_table.columns = df_time_table.columns.droplevel()
    df_time_table = df_time_table.reset_index()
    df_time_table['table_captured_h'] = df_time_table[concept_name_plus].apply(lambda x:datetime(2024,2,1) + x)
    df_time_table['time_table'] = df_time_table.apply(lambda x:x[concept_name_minus] - x[concept_name_plus],axis=1)
    df_time_table['time_table'] = df_time_table['time_table'].apply(lambda x:x.total_seconds()/60)

    df_time_table = df_time_table.sort_values(by=[concept_name_plus])

    fig_3 = px.line(df_time_table, x='table_captured_h',y='time_table', labels={
            "time_table": y_label_line_time_duration,
            "table_captured_h":x_label_line_time_duration
        },
        title=title_line_time_duration)
    # fig.show()

    fig_4 = px.histogram(df_time_table, x="time_table",labels={
                        "time_table": xlabel_histogram,
                    },
                    title=title_histogram)
    fig_4.update_layout(
        yaxis_title="# Occurrencies"
    )

    # fig.show()

    return fig_1, fig_2, fig_3, fig_4


class Simulation_element:
    def __init__(self) -> None:
        self.__id = 0
    
    def get_id(self):
        self.__id+=1
        return self.__id

class Client:
    def __init__(self, env:simpy.Environment, arrival_time: float, client_id:int, table_size:int, order_dessert:int, tables:list[simpy.Resource], waiter: Resource, simulation_element:Simulation_element, log:list[dict]) -> None:
        self.env = env    
        self.arrival_time = arrival_time    
        self.client_id = client_id
        self.table_size = table_size        
        self.order_dessert = order_dessert
        self.tables = tables
        self.waiter = waiter
        self.log = log
        self.simulation_element = simulation_element
        self.id = self.simulation_element.get_id()

        self.name = self.__class__.__name__
        
        # variables
        # variables list that controle whether the client was completely unloaded
        self.unload_complete_list = [env.event() for _ in range(len(self.tables))]

        self.wait_waiter = env.event()
        self.read_menu = env.event()
        
        self.drink_order_ready = env.event()
        self.order_ready = env.event()
        

        self.env.process(self.run_simulation())


    def run_simulation(self):
        # time until the client arrive
        yield self.env.timeout(self.arrival_time)

        # arrive at the parking slot        
        register_log(self.log,self.id,'arrived at the model',self.env.now,entity=self.name)
    
        # wait until get a table
        # if client needs just a regular table, will try to get a regular table first and then will try the big one
        selected_req, selected_req_idx, selected = yield self.env.process(get_resource_from_list(self.env, self.tables))
        if self.table_size == 1:            
            selected_req_idx = 1


        register_log(self.log,self.id,'table captured',self.env.now,entity=self.name,table=selected_req_idx)

        # displace until table and sit
        yield self.env.timeout(triang_values(min_value=1,mode_value=1.5, max_value=2))
        register_log(self.log,self.id,'client sit at the table and start read menu',self.env.now,entity=self.name,table=selected_req_idx)

        
            
        # request the waiter
        # call the waiter to order a drink and read menu
        yield self.env.process(self.call_waiter_to_order(selected_req_idx = selected_req_idx, order_type = 'meal'))

        with self.waiter.request() as req:
            yield req
            register_log(self.log,self.id,'waiter arrived at the table with the menu and take the drink orders',self.env.now,entity=self.name,table=selected_req_idx)

            # waiter getting the drink orders
            if self.table_size == 0:
                yield self.env.timeout(triang_values(1,3,5))
            else:
                yield self.env.timeout(triang_values(2,5,7))

            self.env.process(self.time_to_read_menu())

            # walk until the kitchen
            register_log(self.log,self.id,'waiter go to the kitchen with the drink orders',self.env.now,entity=self.name,table=selected_req_idx)
            yield self.env.timeout(triang_values(1,1.5,2))
            
        # start to process the drink order
        if self.table_size == 0:
            time_preparation = triang_values(10,12,15)
        else:
            time_preparation = triang_values(15,18,20)
        order = Order(self.env, time_preparation=time_preparation, waiter=self.waiter, order_type='drink', order_ready=self.drink_order_ready, id=self.id, log=self.log)

        # wait until everybody has read the menu
        yield self.read_menu

        # call the waiter to order a meal
        yield self.env.process(self.call_waiter_to_order(selected_req_idx = selected_req_idx, order_type = 'meal'))
        yield self.env.timeout(self.time_to_eat(order_type='meal', selected_req_idx = selected_req_idx))
        
        # call the waiter to order a dessert
        if self.order_dessert == 1:
            yield self.env.process(self.call_waiter_to_order(selected_req_idx = selected_req_idx, order_type = 'dessert'))            
            yield self.env.timeout(self.time_to_eat(order_type='dessert', selected_req_idx = selected_req_idx))

        # call the waiter to pay the bill
        yield self.env.process(self.pay_bill(selected_req_idx=selected_req_idx))

        # client leaving the table
        selected_req.release(selected)

        register_log(self.log,self.id,'table released',self.env.now,entity=self.name,table=selected_req_idx)
        
    def time_to_read_menu(self):
        yield self.env.timeout(triang_values(min_value=5,mode_value=7, max_value=15))
        self.read_menu.succeed()
    
    def call_waiter_to_order(self, selected_req_idx: int, order_type:str, read_menu=False):
        register_log(self.log,self.id,'call a waiter',self.env.now,entity=self.name,table=selected_req_idx)

        # meal orders
        with self.waiter.request() as req:
            # waiter getting the orders
            yield req
            waiter_id = self.waiter.count
            register_log(self.log,self.id,'waiter arrived at the table',self.env.now,entity=self.name,table=selected_req_idx, waiter=waiter_id)
            
            # time to get the order
            time_order = self.time_to_get_the_order(order_type=order_type)
            yield self.env.timeout(time_order)

            # walk until the kitchen
            register_log(self.log,self.id,'waiter go to the kitchen with the orders',self.env.now,entity=self.name,table=selected_req_idx)
            yield self.env.timeout(triang_values(1,1.5,2))

        # start to process the mean order
        time_preparation = self.time_to_prepare_the_order(order_type=order_type)

        # start to process the mean order
        order = Order(self.env, time_preparation=time_preparation, waiter=self.waiter, order_type='meal', order_ready=self.order_ready, id=self.id, log=self.log)
        

        # wait until meal orders arrive
        yield self.order_ready
        self.order_ready = self.env.event()
    
    def time_to_get_the_order(self, order_type):
        if order_type == 'meal':
            if self.table_size == 0:
                return triang_values(2,4,7)
            else:
                return triang_values(5,7,10)
        elif order_type == 'drink' or order_type == 'dessert':
            if self.table_size == 0:
                return triang_values(1,3,5)
            else:
                return triang_values(2,5,7)
        else:
            raise ValueError(f'order_type {order_type} doesnt exist.' )

    def time_to_prepare_the_order(self, order_type):
        if order_type == 'meal':
            if self.table_size == 0:
                return triang_values(10,12,15)
            else:
                return triang_values(15,18,20)        
        elif order_type == 'drink' or order_type == 'dessert':
            if self.table_size == 0:
                return triang_values(1,3,5)
            else:
                return triang_values(2,5,7) 
        else:
            raise ValueError(f'order_type {order_type} doesnt exist.' )
    
    def time_to_eat(self, order_type:str, selected_req_idx:int):
        register_log(self.log,self.id,f'table start to eat {order_type}',self.env.now,entity=self.name,table=selected_req_idx)
        if order_type == 'meal':
            if self.table_size == 0:
                return triang_values(15,25,30)
            else:
                return triang_values(25,30,60)
        elif order_type == 'dessert':
            if self.table_size == 0:
                return triang_values(10,20,25)
            else:
                return triang_values(20,25,30)
        else:
            raise ValueError(f'order_type {order_type} doesnt exist.' )
    
    def pay_bill(self, selected_req_idx):
        with self.waiter.request() as req:
            register_log(self.log,self.id,'looking for a waiter',self.env.now,entity=self.name,table=selected_req_idx)
            yield req
            register_log(self.log,self.id,'call waiter to bring the bill',self.env.now,entity=self.name,table=selected_req_idx)
            
        # time to process the bill
        yield self.env.timeout(triang_values(5,7,10))
        register_log(self.log,self.id,'bill ready to be delivery to the table',self.env.now,entity=self.name,table=selected_req_idx)

        with self.waiter.request() as req:
            yield req
            register_log(self.log,self.id,'waiter with the bill',self.env.now,entity=self.name,table=selected_req_idx)

            if self.table_size == 0:
                yield self.env.timeout(triang_values(1,2,3))
            else:
                yield self.env.timeout(triang_values(1,5,7))
            register_log(self.log,self.id,'bill payed',self.env.now,entity=self.name,table=selected_req_idx)
        
        # approaching to leave
        yield self.env.timeout(triang_values(1,2,5))

class Order:
    def __init__(self, env:simpy.Environment, time_preparation:Union[int,float], waiter:Resource, order_type: str, order_ready:simpy.Event, id:int,log:list[dict]) -> None:
        self.env = env
        self.time_preparation = time_preparation
        self.waiter = waiter
        self.order_type = order_type

        self.id = id
        self.log = log
        self.name = self.__class__.__name__

        # variable
        self.order_ready = order_ready

        self.env.process(self.run_simulation())
        
        
    def run_simulation(self):
        yield self.env.timeout(self.time_preparation)
        
        register_log(self.log,self.id,f'order ready - {self.order_type}',self.env.now,entity=self.name)

        with self.waiter.request() as req:
            register_log(self.log,self.id,f'call a waiter - {self.order_type}',self.env.now,entity=self.name)
            yield req
            waiter_id = self.waiter.count
            register_log(self.log,self.id,f'capture a waiter - {self.order_type}',self.env.now,entity=self.name,waiter=waiter_id)
            
            # trip until the table
            yield self.env.timeout(triang_values(.5,1,1.5))
            register_log(self.log,self.id,'waiter arrived at the table',self.env.now,entity=self.name,waiter=waiter_id)

            self.order_ready.succeed()


def run_simulation(env:simpy.Environment, tables: list[Resource], log: list[dict],waiter:Resource, simulation_element:Simulation_element, clients_hr:np.ndarray, number_people_table_profile:np.ndarray, prob_order_dessert:np.ndarray):
    client_id = 0
    for hr,number_hr in enumerate(clients_hr):
        for _ in range(number_hr):
            client_id+=1

            # client will arrive at the drawed hour and in a random minute within the 60 min in an hour
            arrival_time = hr * 60 + np.random.random()*60
            table_size = draw_values_from_profile(arrival_profile_prob=number_people_table_profile)[0]
            order_dessert = draw_values_from_profile(arrival_profile_prob=number_people_table_profile)[0]
            client = Client(env, arrival_time=arrival_time, client_id=client_id, table_size=table_size, order_dessert=order_dessert, tables=tables, waiter=waiter, log=log, simulation_element=simulation_element)


def start_new_simulation(
        df_resource: pd.DataFrame,
        df_location: pd.DataFrame,
        df_total_arrival: pd.DataFrame,
        df_arrival_hr: pd.DataFrame,
        df_number_people_table_profile: pd.DataFrame,
        df_prob_order_dessert: pd.DataFrame,
        df_time_input: pd.DataFrame,
        
    ):
    env = simpy.Environment()

    simulation_element = Simulation_element()
    log = []

    # 10 regular tables and 2 big tables
    tables = [
        simpy.Resource(env, df_location.values.flatten()[0]),
        simpy.Resource(env, df_location.values.flatten()[1])]
    waiter = simpy.Resource(env, df_resource.values[0][0])

    total_clients_day = df_total_arrival.values.flatten()[0]

    # arrival_profile_prob = np.array([0,0,0,0,0,0,0,0,0,0,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,0,0,0,0])
    arrival_profile_prob = df_arrival_hr['% Clients in each hour'].values/100
    clients_hr = arrival_profile_prob * total_clients_day
    clients_hr = (np.ceil(clients_hr)).astype(int)

    # 70% clients with regular table and 30% clients big tables
    # number_people_table_profile = np.array([.7,.3])
    number_people_table_profile = df_number_people_table_profile['Probability Client Tables'].values
    prob_order_dessert = df_prob_order_dessert['Probability Table order dessert'].values

    run_simulation(env=env, tables=tables, log=log, waiter=waiter, simulation_element=simulation_element, clients_hr=clients_hr, number_people_table_profile=number_people_table_profile, prob_order_dessert=prob_order_dessert)

    # run for 1 day
    # env.run(until=24 * 60)
    env.run()

    return log


with st.container(border=True):
    with st.expander("Define the resources"):
        col1, col2, _ = st.columns(3)
        with col1:
            df_resource = st.data_editor(df_resource, disabled=['Resource'])
        with col2:
            df_location = st.data_editor(df_location, disabled=['Resource'])

    with st.expander("Clients arrival setup"):
        col1, col2 = st.columns(2)

        with col1:
            df_total_arrival = st.data_editor(df_total_arrival)
            
            arrival_profile_prob = np.array([0,0,0,0,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,0,0,0,0])
            arrival_profile_labels = [f'{idx} - {idx + 1}' for idx in range(arrival_profile_prob.shape[0])]        

            df_arrival_hr = st.data_editor(df_arrival_hr)
            s = df_arrival_hr['% Clients in each hour'].sum()
            if s != 100:
                st.markdown(f" ## Sum must be 100, but got {s}")
        

            # 70% clients with regular table and 30% clients big tables
            df_number_people_table_profile = st.data_editor(df_number_people_table_profile)

            df_prob_order_dessert = st.data_editor(df_prob_order_dessert)



        with col2:
            fig = px.bar(df_arrival_hr, orientation='h', width=300)
            fig.update_layout(showlegend=False)
            st.write(fig)

    with st.expander("Tasks time duration setup"):
        st.write("Time")
        
        df_time_input = st.data_editor(df_time_input)


btn_run_simulation = st.button("Run simulation")

if btn_run_simulation:
    log = start_new_simulation(
        df_resource=df_resource,
        df_location=df_location,
        df_total_arrival=df_total_arrival,
        df_arrival_hr=df_arrival_hr,
        df_number_people_table_profile=df_number_people_table_profile,
        df_prob_order_dessert=df_prob_order_dessert,
        df_time_input=df_time_input,
    )

    df = pd.DataFrame(log)

    df['time:timestamp'] = df['time:timestamp:number'].apply(lambda x:pd.to_datetime(x,unit='m'))
    viz_count, viz_performance = generate_output_dfg(df=df,entity='Client')
    df['time:timestamp:delta'] = df['time:timestamp:number'].apply(lambda x:pd.to_timedelta(x,unit='m'))

    tab1, tab2 = st.tabs(["Frequency", "Performance"])

    with tab1:
        st.header("Frequency DFG")
        with st.container(height=500):
            viz_count
    
    with tab2:
        st.header("Performance DFG")
        with st.container(height=500):
            viz_performance



    tab1, tab2 = st.tabs(["Table Ocuppacy", "Queuing to get a table"])
    with tab1:
        fig_1, fig_2, fig_3, fig_4 = create_graphs(df=df, concept_name_plus='table captured', concept_name_minus='table released')
        st.header("Table Ocuppacy")
        with st.container(height=500):
            st.write(fig_1)
            st.write(fig_2)
            st.write(fig_3)
            st.write(fig_4)
    with tab2:
        fig_1, fig_2, fig_3, fig_4 = create_graphs(df=df, concept_name_plus='arrived at the model', concept_name_minus='table captured',
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

    # save the scenario state
    st.session_state['scenarios_simulated'][scenario_name] = {
        'df_resource':df_resource,
        'df_location':df_location,
        'df_location':df_location,
        'df_total_arrival':df_total_arrival,
        'df_arrival_hr':df_arrival_hr,
        'df_number_people_table_profile':df_number_people_table_profile,
        'df_prob_order_dessert':df_prob_order_dessert,
        'df_time_input':df_time_input,
        'log':log,
    }