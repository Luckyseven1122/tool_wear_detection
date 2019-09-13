from fastai.tabular import *
from flask import Flask, request
import requests
import os.path


path = ''

export_file_url = 'https://www.dropbox.com/s/vhw90tbhj702qby/tool_wear_7.pkl?dl=1'
export_file_name = 'tool_wear_7.pkl'


def down_load_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)
            
def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        down_load_file(filename, url)
        return True
    return False

download_if_not_exists(export_file_name, export_file_url)

learn = load_learner(path, export_file_name)

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        X1_ActualPosition = float(request.form.get('X1_ActualPosition'))
        X1_ActualVelocity = float(request.form.get('X1_ActualVelocity'))
        X1_ActualAcceleration = float(request.form.get('X1_ActualAcceleration'))
        X1_CommandPosition = float(request.form.get('X1_CommandPosition'))
        X1_CommandVelocity = float(request.form.get('X1_CommandVelocity'))
        X1_CommandAcceleration = float(request.form.get('X1_CommandAcceleration'))
        X1_CurrentFeedback = float(request.form.get('X1_CurrentFeedback'))
        X1_DCBusVoltage = float(request.form.get('X1_DCBusVoltage'))
        X1_OutputCurrent = float(request.form.get('X1_OutputCurrent'))
        X1_OutputVoltage = float(request.form.get('X1_OutputVoltage'))
        X1_OutputPower = float(request.form.get('X1_OutputPower'))
        Y1_ActualPosition = float(request.form.get('Y1_ActualPosition'))
        Y1_ActualVelocity = float(request.form.get('Y1_ActualVelocity'))
        Y1_ActualAcceleration = float(request.form.get('Y1_ActualAcceleration'))
        Y1_CommandPosition = float(request.form.get('Y1_CommandPosition'))
        Y1_CommandVelocity = float(request.form.get('Y1_CommandVelocity'))
        Y1_CommandAcceleration = float(request.form.get('Y1_CommandAcceleration'))
        Y1_CurrentFeedback = float(request.form.get('Y1_CurrentFeedback'))
        Y1_DCBusVoltage = float(request.form.get('Y1_DCBusVoltage'))
        Y1_OutputCurrent = float(request.form.get('Y1_OutputCurrent'))
        Y1_OutputVoltage = float(request.form.get('Y1_OutputVoltage'))
        Y1_OutputPower = float(request.form.get('Y1_OutputPower'))
        Z1_ActualPosition = float(request.form.get('Z1_ActualPosition'))
        Z1_ActualVelocity = float(request.form.get('Z1_ActualVelocity'))
        Z1_ActualAcceleration = float(request.form.get('Z1_ActualAcceleration'))
        Z1_CommandPosition = float(request.form.get('Z1_CommandPosition'))
        Z1_CommandVelocity = float(request.form.get('Z1_CommandVelocity'))
        Z1_CommandAcceleration = float(request.form.get('Z1_CommandAcceleration'))
        Z1_CurrentFeedback = float(request.form.get('Z1_CurrentFeedback'))
        Z1_DCBusVoltage = float(request.form.get('Z1_DCBusVoltage'))
        Z1_OutputCurrent = float(request.form.get('Z1_OutputCurrent'))
        Z1_OutputVoltage = float(request.form.get('Z1_OutputVoltage'))
        S1_ActualPosition = float(request.form.get('S1_ActualPosition'))
        S1_ActualVelocity = float(request.form.get('S1_ActualVelocity'))
        S1_ActualAcceleration = float(request.form.get('S1_ActualAcceleration'))
        S1_CommandPosition = float(request.form.get('S1_CommandPosition'))
        S1_CommandVelocity = float(request.form.get('S1_CommandVelocity'))
        S1_CommandAcceleration = float(request.form.get('S1_CommandAcceleration'))
        S1_CurrentFeedback = float(request.form.get('S1_CurrentFeedback'))
        S1_DCBusVoltage = float(request.form.get('S1_DCBusVoltage'))
        S1_OutputCurrent = float(request.form.get('S1_OutputCurrent'))
        S1_OutputVoltage = float(request.form.get('S1_OutputVoltage'))
        S1_OutputPower = float(request.form.get('S1_OutputPower'))
        S1_SystemInertia = float(request.form.get('S1_SystemInertia'))
        M1_CURRENT_PROGRAM_NUMBER = float(request.form.get('M1_CURRENT_PROGRAM_NUMBER'))
        M1_sequence_number = float(request.form.get('M1_sequence_number'))
        M1_CURRENT_FEEDRATE = float(request.form.get('M1_CURRENT_FEEDRATE'))
        Machining_Process = request.form.get('Machining_Process')
        
        
        inf_df = pd.DataFrame(columns=['X1_ActualPosition', 'X1_ActualVelocity', 
                                                            'X1_ActualAcceleration','X1_CommandPosition', 
                                                            'X1_CommandVelocity', 'X1_CommandAcceleration',
                                                            'X1_CurrentFeedback', 'X1_DCBusVoltage', 'X1_OutputCurrent',
                                                            'X1_OutputVoltage', 'X1_OutputPower', 'Y1_ActualPosition',
                                                            'Y1_ActualVelocity', 'Y1_ActualAcceleration', 'Y1_CommandPosition',
                                                            'Y1_CommandVelocity', 'Y1_CommandAcceleration', 'Y1_CurrentFeedback',
                                                            'Y1_DCBusVoltage', 'Y1_OutputCurrent', 'Y1_OutputVoltage',
                                                            'Y1_OutputPower', 'Z1_ActualPosition', 'Z1_ActualVelocity',
                                                            'Z1_ActualAcceleration', 'Z1_CommandPosition', 'Z1_CommandVelocity',
                                                            'Z1_CommandAcceleration', 'Z1_CurrentFeedback', 'Z1_DCBusVoltage',
                                                            'Z1_OutputCurrent', 'Z1_OutputVoltage', 'S1_ActualPosition',
                                                            'S1_ActualVelocity', 'S1_ActualAcceleration', 'S1_CommandPosition',
                                                            'S1_CommandVelocity', 'S1_CommandAcceleration', 'S1_CurrentFeedback',
                                                            'S1_DCBusVoltage', 'S1_OutputCurrent', 'S1_OutputVoltage',
                                                            'S1_OutputPower', 'S1_SystemInertia', 'M1_CURRENT_PROGRAM_NUMBER',
                                                            'M1_sequence_number', 'M1_CURRENT_FEEDRATE', 'Machining_Process'])
        
        inf_df.loc[0] = [X1_ActualPosition,X1_ActualVelocity,X1_ActualAcceleration,X1_CommandPosition,
                         X1_CommandVelocity,X1_CommandAcceleration,X1_CurrentFeedback,X1_DCBusVoltage,
                         X1_OutputCurrent,X1_OutputVoltage,X1_OutputPower,Y1_ActualPosition,Y1_ActualVelocity,
                         Y1_ActualAcceleration,Y1_CommandPosition,Y1_CommandVelocity,Y1_CommandAcceleration,
                         Y1_CurrentFeedback,Y1_DCBusVoltage,Y1_OutputCurrent,Y1_OutputVoltage,Y1_OutputPower,
                         Z1_ActualPosition,Z1_ActualVelocity,Z1_ActualAcceleration,Z1_CommandPosition,
                         Z1_CommandVelocity,Z1_CommandAcceleration,Z1_CurrentFeedback,Z1_DCBusVoltage,Z1_OutputCurrent,
                         Z1_OutputVoltage,S1_ActualPosition,S1_ActualVelocity,S1_ActualAcceleration,S1_CommandPosition,
                         S1_CommandVelocity,S1_CommandAcceleration,S1_CurrentFeedback,S1_DCBusVoltage,S1_OutputCurrent,
                         S1_OutputVoltage,S1_OutputPower,S1_SystemInertia,M1_CURRENT_PROGRAM_NUMBER,M1_sequence_number,M1_CURRENT_FEEDRATE,Machining_Process]
        
        answer = learn.predict(inf_df.iloc[0])
        
        return '''<h1>The tool is inferred to be: {}</h1>'''.format(answer)
    
    return '''<form method="POST">
                X1_ActualPosition: <input type="number" name="X1_ActualPosition" value="198.0"><br>
                X1_ActualVelocity: <input type="number" name="X1_ActualVelocity" value="0.0"><br>
                X1_ActualAcceleration: <input type="number" name="X1_ActualAcceleration" value="0.0"><br>
                X1_CommandPosition: <input type="number" name="X1_CommandPosition" value="198.0"><br>
                X1_CommandVelocity: <input type="number" name="X1_CommandVelocity" value="0.0"><br>
                X1_CommandAcceleration: <input type="number" name="X1_CommandAcceleration" value="0.0"><br>
                X1_CurrentFeedback: <input type="number" name="X1_CurrentFeedback" value="0.18"><br>
                X1_DCBusVoltage: <input type="number" name="X1_DCBusVoltage" value="0.0207"><br>
                X1_OutputCurrent: <input type="number" name="X1_OutputCurrent" value="329.0"><br>
                X1_OutputVoltage: <input type="number" name="X1_OutputVoltage" value="2.77"><br>
                X1_OutputPower: <input type="number" name="X1_OutputPower" value="-1.42e-06"><br>
                Y1_ActualPosition: <input type="number" name="Y1_ActualPosition" value="158.0"><br>
                Y1_ActualVelocity: <input type="number" name="Y1_ActualVelocity" value="-0.025"><br>
                Y1_ActualAcceleration: <input type="number" name="Y1_ActualAcceleration" value="-6.25"><br>
                Y1_CommandPosition: <input type="number" name="Y1_CommandPosition" value="158.0"><br>
                Y1_CommandVelocity: <input type="number" name="Y1_CommandVelocity" value="0.0"><br>
                Y1_CommandAcceleration: <input type="number" name="Y1_CommandAcceleration" value="0.0"><br>
                Y1_CurrentFeedback: <input type="number" name="Y1_CurrentFeedback" value="0.539"><br>
                Y1_DCBusVoltage: <input type="number" name="Y1_DCBusVoltage" value="0.0167"><br>
                Y1_OutputCurrent: <input type="number" name="Y1_OutputCurrent" value="328.0"><br>
                Y1_OutputVoltage: <input type="number" name="Y1_OutputVoltage" value="1.84"><br>
                Y1_OutputPower: <input type="number" name="Y1_OutputPower" value="6.429999999999999e-07"><br>
                Z1_ActualPosition: <input type="number" name="Z1_ActualPosition" value="119.0"><br>
                Z1_ActualVelocity: <input type="number" name="Z1_ActualVelocity" value="0.0"><br>
                Z1_ActualAcceleration: <input type="number" name="Z1_ActualAcceleration" value="0.0"><br>
                Z1_CommandPosition: <input type="number" name="Z1_CommandPosition" value="119.0"><br>
                Z1_CommandVelocity: <input type="number" name="Z1_CommandVelocity" value="0.0"><br>
                Z1_CommandAcceleration: <input type="number" name="Z1_CommandAcceleration" value="0.0"><br>
                Z1_CurrentFeedback: <input type="number" name="Z1_CurrentFeedback" value="0.0"><br>
                Z1_DCBusVoltage: <input type="number" name="Z1_DCBusVoltage" value="0.0"><br>
                Z1_OutputCurrent: <input type="number" name="Z1_OutputCurrent" value="0.0"><br>
                Z1_OutputVoltage: <input type="number" name="Z1_OutputVoltage" value="0.0"><br>
                S1_ActualPosition: <input type="number" name="S1_ActualPosition" value="-361.0"><br>
                S1_ActualVelocity: <input type="number" name="S1_ActualVelocity" value="0.001"><br>
                S1_ActualAcceleration: <input type="number" name="S1_ActualAcceleration" value="0.25"><br>
                S1_CommandPosition: <input type="number" name="S1_CommandPosition" value="-361.0"><br>
                S1_CommandVelocity: <input type="number" name="S1_CommandVelocity" value="0.0"><br>
                S1_CommandAcceleration: <input type="number" name="S1_CommandAcceleration" value="0.0"><br>
                S1_CurrentFeedback: <input type="number" name="S1_CurrentFeedback" value="0.524"><br>
                S1_DCBusVoltage: <input type="number" name="S1_DCBusVoltage" value="2.74e-19"><br>
                S1_OutputCurrent: <input type="number" name="S1_OutputCurrent" value="329.0"><br>
                S1_OutputVoltage: <input type="number" name="S1_OutputVoltage" value="0.0"><br>
                S1_OutputPower: <input type="number" name="S1_OutputPower" value="6.96e-07"><br>
                S1_SystemInertia: <input type="number" name="S1_SystemInertia" value="12.0"><br>
                M1_CURRENT_PROGRAM_NUMBER: <input type="number" name="M1_CURRENT_PROGRAM_NUMBER" value="1.0"><br>
                M1_sequence_number: <input type="number" name="M1_sequence_number" value="0.0"><br>
                M1_CURRENT_FEEDRATE: <input type="number" name="M1_CURRENT_FEEDRATE" value="50.0"><br>
                Machining_Process: <select name="Machining_Process">
                <option value="Starting">Starting</option>
                <option value="Prep">Prep</option>
                <option value="Layer 1 Up">Layer 1 Up</option>
                <option value="Layer 1 Down">Layer 1 Down</option>
                <option value="Repositioning">Repositioning</option>
                <option value="Layer 2 Up">Layer 2 Up</option>
                <option value="Layer 2 Down">Layer 2 Down</option>
                <option value="Layer 3 Up">Layer 3 Up</option>
                <option value="Layer 3 Down">Layer 3 Down</option>
                <option value="Repositioning">Repositioning</option>
                <option value="end">end</option>
                <option value="End">End</option>
                </select><br>
                <input type="submit" value="Submit"><br>
              </form>'''

