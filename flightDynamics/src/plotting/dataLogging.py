import os

import numpy as np
import pandas as pd


class Logger:
    def __init__(self):
        self.time=[]
        self.states=[]
        self.controls=[]
        self.commands=[]

    def log(self,time,state,controls, commands=np.array([[None, None, None]]).T):
        self.time.append(time)
        self.states.append(state.ravel().copy())
        self.controls.append(controls.ravel().copy())
        self.commands.append(commands.ravel().copy())

    def export(self):
        time=np.asarray(self.time,dtype=np.float32)
        states=np.asarray(self.states,dtype=np.float32)
        controls=np.asarray(self.controls,dtype=np.float32)
        commands=np.asarray(self.commands,dtype=np.float32)
        state_columns=["pn","pe","pd","u","v","w","phi","theta","psi","p","q","r"]
        control_columns=["del_e","del_t","del_a","del_r"]
        command_columns=['pc','qc','rc']
        state_frame=pd.DataFrame(states,columns=state_columns)
        control_frame=pd.DataFrame(controls,columns=control_columns)
        command_frame=pd.DataFrame(commands,columns=command_columns)
        data_frame=pd.concat([state_frame,control_frame,command_frame],axis=1)
        data_frame.insert(0,"t",time)
        os.makedirs("out",exist_ok=True)
        data_frame.to_parquet(os.path.join("out","output.parquet"),index=False)
        np.savez_compressed(
            os.path.join("out","output.npz"),t=time,x=states,u=controls,c=commands,
        )
