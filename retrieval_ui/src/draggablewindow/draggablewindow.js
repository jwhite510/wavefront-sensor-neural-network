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

  // Bind it to a component
    return <animated.div className='movingcube' {...bind()} style={{
      x:springprops.x,
      y:springprops.y,
      touchAction:'none'
      }}>

      <p>
      retrieval_type:
      {props.retrieval_type}
      </p><p>
      diffraction_type:
      {props.diffraction_type}
      </p>

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
