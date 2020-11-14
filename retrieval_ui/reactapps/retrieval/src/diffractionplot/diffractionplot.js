import Plot from 'react-plotly.js';

export default function DiffractionPlot(props) {
  var trace1 = {
    z: props.inputdata.xintensity,
    xaxis: 'x3',
    yaxis: 'y3',
    type: 'heatmap',
    colorbar:{
      thickness:10,
      x:1.0,
      y:0.75,
      len:0.4,
    },
  };

  var trace2 = {
    z: props.inputdata.diffraction,
    xaxis: 'x4',
    yaxis: 'y4',
    type: 'heatmap',
    colorbar:{
      thickness:10,
      x:0.0,
      y:0.80,
      len:0.4,
    },
  };

  var trace3 = {
    z: props.inputdata.xphase,
    xaxis: 'x2',
    yaxis: 'y2',
    type: 'heatmap',
    colorbar:{
      thickness:10,
      x:1.0,
      y:0.25,
      len:0.4,
    },
  };

  var data = [
    trace1,
    trace2,
    trace3
  ];

  var layout = {
    colorbar:false,
    margin:{l:30,r:30,t:30,b:30},
    width: 400,
    height: 300,
    title: 'title',
    showlegend:false,
    // right bottom
    yaxis2: { domain: [0.0, 0.49], anchor: 'x2',visible:false},
    xaxis2: { domain: [0.51, 1], anchor: 'y2',visible:false},
    // right top
    yaxis3: { domain: [0.51, 1], anchor: 'x3',visible:false},
    xaxis3: { domain: [0.51, 1], anchor: 'y3',visible:false},
    // left
    xaxis4: { domain: [0, 0.5], anchor: 'y4',visible:false},
    yaxis4: { domain: [0, 1], anchor: 'x4' ,visible:false},
  };
  return <Plot
		data={data}
		layout={layout}
		config={{displayModeBar:false}}
		// onInitialized={(figure) => this.setState(figure)}
		// onUpdate={(figure) => this.setState(figure)}
	      />

}
