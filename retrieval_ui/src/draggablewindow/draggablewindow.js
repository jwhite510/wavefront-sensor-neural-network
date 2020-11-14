import React from 'react'
import ReactDOM from 'react-dom'
import { useSpring, animated, to } from 'react-spring'
import { useGesture } from 'react-use-gesture'
import './styles.css'
import { useDrag } from 'react-use-gesture'
import DiffractionPlot from '../diffractionplot/diffractionplot'
// document.addEventListener('gesturestart', e => e.preventDefault())
// document.addEventListener('gesturechange', e => e.preventDefault())

function getCookie(name) {
  var cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    var cookies = document.cookie.split(';');
    for (var i = 0; i < cookies.length; i++) {
      var cookie = (cookies[i]).trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
	cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
	break;
      }
    }
  }
  return cookieValue;
}

function Xbutton(props){
  return <animated.div
    style={{transform:'rotate(45deg)',color:'red',fontSize:'30px'}}
    onClick={() => {
      props.settoggle(toggle=>false)
      setTimeout(function(){
	props.removewindow(props.magicthing)
      },3000)
      // reset=true;

    }}

  >
    +
  </animated.div>
}

export default function PullRelease(props) {
  const [inputdata, setinputdata] = React.useState({
		      diffraction:[[1, 1, 1], [1,1,1], [1,1,1]],
		      xintensity:[[2, 2, 2], [2,2,2], [2,2,2]],
		      xphase:[[3, 3, 3], [3,3,3], [3,3,3]],
		      })

  const [springprops, set_sp] = useSpring(() => ({ x: 0, y: 0 }))
  const [staticprops, set_st] = React.useState({ x_start: 0, y_start: 0, isPinching:false,isDragging:false})
  const [parameters,setparameters]=React.useState({magnitude:2,order:3})

  // Set the drag hook and define component movement based on gesture data
  const bind = useGesture(
    {
      onDrag:({event,first,last,movement:[dx,dy]})=>{
	if(!staticprops.isPinching){
	  console.log("dragging");
	  if(first){
	    console.log("first drag");
	    set_st({
	      x_start: springprops.x.get(),
	      y_start: springprops.y.get(),
	      isPinching:staticprops.isPinching
	    })
	    // console.log("event.offsetX =>", event.offsetX);
	    // console.log("event.offsetY =>", event.offsetY);
	    // determine position of drag
	  }
	  else if(last){
	    set_st({
	      x_start: springprops.x.get(),
	      y_start: springprops.y.get(),
	      isPinching:staticprops.isPinching
	    })
	  }
	  else{
	    set_sp({
	      x: staticprops.x_start+dx,
	      y: staticprops.y_start+dy,
	    })
	  }
	}
      },
      onPinch:({da:[d,a],origin,event,first,last,movement:[dx,dy]})=>{
	if(first){
	  set_st({
	    x_start: staticprops.x_start,
	    y_start: staticprops.y_start,
	    isPinching:true,
	  })
	}else if(last){
	  set_st({
	    x_start: staticprops.x_start,
	    y_start: staticprops.y_start,
	    isPinching:false
	  })
	}else{
	  // console.log("event =>", event);
	  console.log("origin =>", origin);
	  console.log("pinching");
	  console.log("d =>", d);
	  console.log("a =>", a);
	}
      }

    }
  )

  const [toggle, settoggle] = React.useState(false)
  // run only on creation
  React.useEffect(()=>{
      settoggle(toggle=>true)
  },[])

  const props2 = useSpring({opacity:toggle?1:0})
  const props3 = useSpring({transform:toggle?'scale(1)':'scale(0)'})
  // Bind it to a component
    return <animated.div className='movingcube'  style={{
      x:springprops.x,
      y:springprops.y,
      touchAction:'none',
      opacity:props2.opacity,
      transform:props3.transform
      }}
    >
      <animated.div {...bind()} className='windowbar'>
	<animated.table style={{width:'100%',height:'100%'}}>
	  <animated.tbody>
	    <animated.tr>
	      <animated.td style={{width:'15%'}}>

		<Xbutton {...props} settoggle={settoggle}/>

	      </animated.td>
	      <animated.td style={{color:'white',fontSize:'30px'}}>{props.diffraction_type} - {props.retrieval_type} </animated.td>
	    </animated.tr>
	  </animated.tbody>
	</animated.table>


      </animated.div>
      <animated.table style={{position:'absolute',top:'10%',width:'100%',height:'90%'}}>
	<animated.tbody>

	  <animated.tr>
	    <animated.td>

	      <animated.button onClick={()=>

	      {console.log('set data')
		console.log('setparameters');
		// setparameters({magnitude:1.337})
		var x = new Array(10)
		for(var i=0; i < x.length; i++){
		  x[i]= new Array(10)
		  for(var j=0; j < x[i].length; j++){
		    x[i][j]= Math.random()
		  }
		}

		setinputdata({
		  diffraction:x,
		  xintensity:[[0, 0, 0], [0,0,0], [0,0,0]],
		  xphase:[[0, 0, 0], [0,0,0], [0,0,0]],
		})

	      }}>Click Me</animated.button>

	      <animated.div className="slidecontainer">

		<input type="number" min="0" max="15" value={parameters.order} onChange={
		  (event)=>{
		    setparameters(params=>{
		      let olsparams = params;
		      olsparams.order = event.target.value;

		      // send to server
		    return fetch('retrieve', {
		      method:'POST',
		      headers: {
			'content-type': 'application/json',
			'X-CSRFToken':getCookie('csrftoken')
		      },
		      body: JSON.stringify({magnitude:olsparams.magnitude,
					    order:olsparams.order})
		      // body: undefined
		    }).then(function(res) {
		      console.log('response received')
		      return res.json();
		    }).then(function(details) {
		      console.log('details: ',details)
		    })


		      return {magnitude:olsparams.magnitude,
			      order:olsparams.order}
		    })

		  }

		}/>
		<input type="range" value={parameters.magnitude} step="0.02" min="-6" max="6" onChange={
		  (event)=>{
		    setparameters(params=>{
		      let olsparams = params;
		      olsparams.magnitude = event.target.value;

		      // send to server


		      return {magnitude:olsparams.magnitude,
			      order:olsparams.order}
		    })

		  }
		  }
		  style={{width:'100%'}}
		  className="slider" id="myRange"/>
		<p>{parameters.magnitude}</p>



	      </animated.div>

	    </animated.td>
	  </animated.tr>
	  <animated.tr>
	    <animated.td>
	      <DiffractionPlot inputdata={inputdata} setinputdata={setinputdata}/>
	    </animated.td>
	  </animated.tr>
	  <animated.tr>
	    <animated.td>
	      <DiffractionPlot inputdata={inputdata} setinputdata={setinputdata}/>
	    </animated.td>
	  </animated.tr>

	</animated.tbody>
      </animated.table>

    </animated.div>
}


// function App() {
//   return (
//     <div>
//       <PullRelease/>
//     </div>
//   )
// }

// ReactDOM.render(<App />, document.getElementById('root'))
