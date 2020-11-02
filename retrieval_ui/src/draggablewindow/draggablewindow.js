import React from 'react'
import ReactDOM from 'react-dom'
import { useSpring, animated, to } from 'react-spring'
import { useGesture } from 'react-use-gesture'
import './styles.css'
import { useDrag } from 'react-use-gesture'
// document.addEventListener('gesturestart', e => e.preventDefault())
// document.addEventListener('gesturechange', e => e.preventDefault())

export default function PullRelease(props) {
  const [springprops, set_sp] = useSpring(() => ({ x: 0, y: 0 }))
  const [staticprops, set_st] = React.useState({ x_start: 0, y_start: 0, isPinching:false,isDragging:false})

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
    return <animated.div className='movingcube' {...bind()} style={{
      x:springprops.x,
      y:springprops.y,
      touchAction:'none',
      opacity:props2.opacity,
      transform:props3.transform
      }}
    >
      <animated.table style={{width:'100%',height:'100%'}}>
	<animated.tbody>

	  <animated.tr>
	    <animated.td>

	      <animated.div 
		style={{transform:'rotate(45deg)'}}
		onClick={() => {
		settoggle(toggle=>false)
		setTimeout(function(){
		  props.removewindow(props.magicthing)
		},3000)
		// reset=true;

		}}

	      >
	      +
	      </animated.div>

	    </animated.td>
	    <animated.td>
	    in a table
	    </animated.td>
	  </animated.tr>
	  <animated.tr>
	    <animated.td>
	      {props.retrieval_type}
	    </animated.td>
	    <animated.td>
	      {props.diffraction_type}
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
