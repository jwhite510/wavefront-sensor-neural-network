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
    this.updatewindows=this.updatewindows.bind(this)
  }

  updatewindows(retrieval_type,diffraction_type){
    console.log('updatewindows called');
    this.setState( {
      draggableboxes:this.state.draggableboxes.concat(<PullRelease retrieval_type={retrieval_type}
	diffraction_type={diffraction_type}
	key={this.state.draggableboxes.length}/>)
      })
  }
  render(){
    return (
      <div className="App">

	<img src={logo} className="App-logo" alt="logo" />

	<div style={{backgroundColor:'',width:'300px',height:'300px'}}>
	  <Opensbutton updatewindows={this.updatewindows}/>
	</div>
	<div>
	  {this.state.draggableboxes}
	</div>

      </div>
    );
  }

}

export default App;
