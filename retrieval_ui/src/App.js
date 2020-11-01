import logo from './logo.svg';
import Opensbutton from './opensbutton/opensbutton'
import PullRelease from './draggablewindow/draggablewindow'
import './App.css';

function App() {
  return (
    <div className="App">

        <img src={logo} className="App-logo" alt="logo" />

      <div style={{backgroundColor:'',width:'300px',height:'300px'}}>
      <Opensbutton/>
      </div>
      <div>
	<PullRelease/>
      </div>

    </div>
  );
}

export default App;
