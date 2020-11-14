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
    this.removewindow=this.removewindow.bind(this)
  }

  updatewindows(retrieval_type,diffraction_type){
    console.log('updatewindows called');
    const aspecialkey=Math.random().toString(36).substring(7)
    this.setState( {
      draggableboxes:this.state.draggableboxes.concat(<PullRelease retrieval_type={retrieval_type}
	diffraction_type={diffraction_type}
	key={aspecialkey}
	magicthing={aspecialkey}
	removewindow={this.removewindow}
	/>)
      })
  }
  removewindow(key){
    console.log('remove windows called');
    console.log("key =>", key);
    // debugger;
    for(var i=0; i < this.state.draggableboxes.length; i++){
      if(this.state.draggableboxes[i].key==key){
	this.state.draggableboxes.splice(i,1)
	break;
      }
    }
    this.setState({draggableboxes:this.state.draggableboxes})
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
