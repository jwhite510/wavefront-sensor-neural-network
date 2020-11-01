import React from 'react'
import logo from './logo.svg';
import Opensbutton from './opensbutton/opensbutton'
import PullRelease from './draggablewindow/draggablewindow'
import './App.css';

function AddThing(props){
  console.log('AddThing called');
  console.log("props.list_of_things =>", props.list_of_things);
  // console.log("props.set_list_of_things =>", props.set_list_of_things);
  console.log("props.list_of_things.length =>", props.list_of_things.length);
  // console.log("props =>", props);
  // debugger;
  props.set_list_of_things([props.list_of_things.concat(<PullRelease 
    key={props.list_of_things.length} />)
  ])
}

function App() {
  const [list_of_things,set_list_of_things]=React.useState([]);

  React.useEffect(()=>{
    console.log('useEffect called');
    // set_list_of_things([<div>hello</div>])
    // set_list_of_things([list_of_things.concat(<div>hello</div>)])

  },[])

  return (
    <div className="App">

        <img src={logo} className="App-logo" alt="logo" />

      <div style={{backgroundColor:'',width:'300px',height:'300px'}}>
      <Opensbutton/>
      </div>
      <div>
	<PullRelease/>
      </div>
      {list_of_things}
      <button onClick={()=>AddThing({list_of_things,set_list_of_things})}>
	click me
      </button>


    </div>
  );
}

export default App;
