import React, { useState, useRef } from 'react'
import { animated } from 'react-spring'
import { render } from 'react-dom'
import { useTransition, useSpring, useChain, config } from 'react-spring'
import { Global, Container, Item } from './styles'
import data from './data'
import data2 from './data2'
import '../index.css';

export default function Opensbutton() {
  const [open, set] = useState(false)
  const [open2, set2] = useState(false)

  const springRef = useRef()
  const { radius, size, opacity, ...rest } = useSpring({
    ref: springRef,
    config: config.stiff,
    from: {
      size: '10%',
      background: 'green' ,
      radius: '50px'
    },
    to: {
      size: open ? '30%' : '10%',
      background: open ? 'white' : 'green' ,
      radius: open ? '5px' : '50px'
    }
  })

  const transRef = useRef()
  const transitions = useTransition(open ? data : [], item => item.name, {
    ref: transRef,
    unique: true,
    trail: 400 / data.length,
    from: { opacity: 0, transform: 'scale(0)' },
    enter: { opacity: 1, transform: 'scale(1)' },
    leave: { opacity: 0, transform: 'scale(0)' }
  })

  const transRef2 = useRef()
  const secondmenutransitions = useTransition(open2 ? data2 : [], item => item.name, {
    ref: transRef2,
    unique: true,
    trail: 400 / data2.length,
    from: { opacity: 0, transform: 'scale(0)' },
    enter: { opacity: 1, transform: 'scale(1)' },
    leave: { opacity: 0, transform: 'scale(0)' }

  })

  // This will orchestrate the two animations above, comment the last arg and it creates a sequence
  useChain(open ? [springRef, transRef, transRef2] : [transRef])
  useChain(open2 ? [transRef2] : [transRef2])

  return (
    <>
      <Global />
      <Container style={{ ...rest, width: size, height: size, borderRadius: radius}} onClick={() => 
	{
	  console.log('container clicked')
	  if(!open){
	    set(open=>true)
	  }
	  // set(open => !open)
	}
	}>
	<animated.div className="centered">
	  +
	</animated.div>

        {transitions.map(({ item, key, props }) => (
          <Item key={key} style={{ ...props, background: item.css }}
	      onClick={()=>{

		console.log('open : '+item.text); 
		console.log('open second menu');
		set(open=>false)
		set2(open2=>true)

	    }}
	  >
	    <animated.div
	      className="centered">
	      {item.text}
	    </animated.div>
	  </Item>
        ))}
        {secondmenutransitions.map(({ item, key, props }) => (
          <Item key={key} style={{ ...props, background: item.css }}
	      onClick={()=>{

		console.log('open : '+item.text); 
		// set(open=>false)

	    }}
	  >
	    <animated.div
	      className="centered">
	      {item.text}
	    </animated.div>
	  </Item>
        ))}






      </Container>
    </>
  )
}
