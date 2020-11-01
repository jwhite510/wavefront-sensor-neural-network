import React from 'react';
import logo from './logo.svg';
import Opensbutton from './opensbutton/opensbutton'
import PullRelease from './draggablewindow/draggablewindow'
import './App.css';

class App extends React.Component {
  constructor(props){
    super(props)
    this.state={
      draggableboxes:[]
    }
    this.updatestate=this.updatestate.bind(this)
  }

  updatestate(){
    console.log('updatestate called');
    this.setState( {
      draggableboxes:this.state.draggableboxes.concat(<PullRelease key={this.state.draggableboxes.length}/>)
      })
  }
  render(){
    return (
      <div className="App">

	<img src={logo} className="App-logo" alt="logo" />

	<div style={{backgroundColor:'',width:'300px',height:'300px'}}>
	  <Opensbutton/>
	</div>
	<div>
	  <PullRelease/>
	</div>
	<div>
	  <button onClick={this.updatestate}>CLICK ME</button>
	  {this.state.draggableboxes}
	</div>

      </div>
    );
  }

}

export default App;
