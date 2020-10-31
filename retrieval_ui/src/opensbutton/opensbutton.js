import React, { useState, useRef } from 'react'
import { animated } from 'react-spring'
import { render } from 'react-dom'
import { useTransition, useSpring, useChain, config } from 'react-spring'
import { Global, Container, Item } from './styles'
import data from './data'
import data2 from './data2'
import '../index.css';

export default function Opensbutton() {
  const [open0, set0] = useState(false)
  const [open1, set1] = useState(false)
  const [open2, set2] = useState(false)
  const [ret_type, set3] = useState({val:""})

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
      size: open0 ? '30%' : '10%',
      background: open0 ? 'white' : 'green' ,
      radius: open0 ? '5px' : '50px'
    }
  })

  const transRef = useRef()
  const transitions = useTransition(open1 ? data : [], item => item.name, {
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
  useChain(open0 ? [springRef] : [springRef])
  useChain(open1 ? [transRef] : [transRef])
  useChain(open2 ? [transRef2] : [transRef2])

  return (
    <>
      <Global />
      <Container style={{ ...rest, width: size, height: size, borderRadius: radius}} onClick={() => 
	{
	  // console.log('container clicked');
	  if(!open0){
	    set0(open0=>true)
	    set1(open1=>true)
	  }
	}
	}>
	<animated.div className="centered">
	  +
	</animated.div>

	<div style={{position:'absolute',width:'100%',height:'100%',pointerEvents:open1?'':'none'}}>
	  {transitions.map(({ item, key, props }) => (
	    <Item key={key} style={{ ...props, background: item.css }}
	      onClick={()=>{
		const original_color=item.css
		item.css='red'
		setTimeout(function(){
		  item.css=original_color;
		},900)
		set1(open1=>false)
		setTimeout(function(){
		  set2(open2=>true)
		},0)

		set3(ret_type=>({val:ret_type.val+item.text+" "}))
		// console.log("ret_type =>", ret_type);

	      }}
	    >
	      <animated.div
		className="centered">
		{item.text}
	      </animated.div>
	    </Item>

	  ))}
	</div>

	<div style={{position:'absolute', width:'100%',height:'100%',pointerEvents:open2?'':'none'}}>
	  {secondmenutransitions.map(({ item, key, props }) => (
	    <Item key={key} style={{ ...props, background: item.css }}
	      onClick={()=>{
		const original_color=item.css
		item.css='red'
		setTimeout(function(){
		  item.css=original_color
		},900)
		setTimeout(function(){
		  set0(open0=>false)
		},900)
		set2(open2=>false)
		set3(ret_type=>({val:ret_type.val+item.text+" "}))
		console.log(item.text, ret_type.val);
		set3(ret_type=>({val:""}))
		// console.log('open : '+item.text); 
	      }}
	    >
	      <animated.div
		className="centered">
		{item.text}
	      </animated.div>
	    </Item>
	  ))}
	</div>






      </Container>
    </>
  )
}
