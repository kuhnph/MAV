import os

import numpy as np
import pandas as pd


class Logger:
    def __init__(self):
        self.time=[]
        self.states=[]
        self.controls=[]
        self.commands=[]
        self.wind=[]
        self.aero=[]

    def log(self,time,state,controls, commands=np.array([[None, None, None, None, None, None]]).T, wind=np.array([[None, None, None]]).T,aero=np.array([[None, None]]).T):
        self.time.append(time)
        self.states.append(state.ravel().copy())
        self.controls.append(controls.ravel().copy())
        self.commands.append(commands.ravel().copy())
        self.wind.append(wind.ravel().copy())
        self.aero.append(aero.ravel().copy())

    def export(self):
        if not self.time:
            print("No telemetry samples to export.")
            return
        time=np.asarray(self.time,dtype=np.float32)
        states=np.asarray(self.states,dtype=np.float32)
        controls=np.asarray(self.controls,dtype=np.float32)
        commands=np.asarray(self.commands,dtype=np.float32)
        wind=np.asarray(self.wind,dtype=np.float32)
        aero=np.asarray(self.aero,dtype=np.float32)
        state_columns=["pn","pe","pd","u","v","w","phi","theta","psi","p","q","r"]
        control_columns=["del_e","del_t","del_a","del_r"]
        command_columns=['p_c','q_c','r_c','phi_c', 'theta_c', 'psi_c']
        wind_columns=['uw','vw','ww']
        aero_columns=['alpha','beta']
        state_frame=pd.DataFrame(states,columns=state_columns)
        control_frame=pd.DataFrame(controls,columns=control_columns)
        command_frame=pd.DataFrame(commands,columns=command_columns)
        wind_frame=pd.DataFrame(wind,columns=wind_columns)
        aero_frame=pd.DataFrame(aero,columns=aero_columns)
        data_frame=pd.concat([state_frame,control_frame,command_frame,wind_frame,aero_frame],axis=1)
        data_frame.insert(0,"t",time)
        os.makedirs("out",exist_ok=True)
        data_frame.to_parquet(os.path.join("out","output.parquet"),index=False)
        np.savez_compressed(
            os.path.join("out","output.npz"),t=time,x=states,u=controls,c=commands,w=wind,a=aero
        )
