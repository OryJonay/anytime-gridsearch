<template>
  <v-layout row wrap>
    <v-flex offset-md3 md5>
      <br>
      <h6 hidden>{{ current_grid }}</h6>
      <v-select :on-change="updateSelection" :options="selections" :value.sync="selected"></v-select>
      <br>
      <svg></svg>
    </v-flex>
  </v-layout>
</template>
<script>
import axios from 'axios'
import vSelect from 'vue-select'
import * as d3 from 'd3'

export default {
  components: {
    vSelect
  },
  computed: {
    current_grid () {
      var newUuid = this.$store.state.grid
      if (newUuid !== this.uuid) {
        this.uuid = newUuid
        this.selections = []
        this.selected = ''
        this.raw_data = []
        this.fillData()
      }
      return this.$store.state.grid
    }
  },
  data () {
    return {
      raw_data: [],
      selected: '',
      selections: [],
      uuid: ''
    }
  },
  methods: {
    updateSelection (val) {
      if (this.selections.length === 0) {
        this.fillData()
        return
      }
      this.selected = val
      this.processData(this.raw_data, val)
    },
    processData (retData, val) {
      if (this.selections.length === 0) {
        this.selections = Object.keys(retData[0]['params'])
      }
      d3.selectAll('g > *').remove()
      var d3Data = retData.map(function (e) { return Object.assign({}, e.params, { 'score': e.scores.find(function (elm) { return elm.scorer === 'score' }).score }) })
      var margin = {top: 20, right: 20, bottom: 30, left: 40}
      var width = 960 - margin.left - margin.right
      var height = 760 - margin.top - margin.bottom
      var xValue = function (d) { return d[val] } // data -> value
      var xScale = typeof (d3Data[0][val]) === 'number' ? d3.scaleLinear().range([0, width]).domain(d3.extent(d3Data.map(function (i) { return i[val] })))
        : d3.scaleOrdinal().range([0, width]).domain(d3.extent(d3Data.map(function (i) { return i[val] }))) // value -> display
      var xMap = function (d) { return xScale(xValue(d)) } // data -> display
      var xAxis = d3.axisBottom(xScale)
      var yValue = function (d) { return d.score } // data -> value
      var yScale = d3.scaleLinear().range([height, 0]).domain([0, 1]) // value -> display
      var yMap = function (d) { return yScale(yValue(d)) } // data -> display
      var yAxis = d3.axisLeft(yScale)
      // setup fill color
      var cValue = function (d) { return d.score }
      var color = d3.scaleLinear().domain([0, 1]).range(['brown', 'steelblue'])
      // add the graph canvas to the body of the webpage
      var svg = d3.select('svg').attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
      // add the tooltip area to the webpage
      var tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0)
      svg.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(0,' + height + ')')
        .call(xAxis)
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', height + 30)
        .style('text-anchor', 'middle')
        .text(val)
      svg.append('g')
        .attr('class', 'y axis')
        .call(yAxis)
      svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', 0 - margin.left)
        .attr('x', 0 - (height / 2))
        .attr('dy', '0.71em')
        .style('text-anchor', 'middle')
        .text('Accuracy %')
      svg.selectAll('.dot')
        .data(d3Data)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('r', 7)
        .attr('cx', xMap)
        .attr('cy', yMap)
        .style('fill', function (d) { return color(cValue(d)) })
        .on('mouseover', function (d) {
          tooltip.transition()
            .duration(200)
            .style('opacity', 0.9)
          tooltip.html(Object.entries(d).map(function (e) { return e.join(' = ') }).join('<br>'))
            .style('left', (d3.event.pageX + 5) + 'px')
            .style('top', (d3.event.pageY - 28) + 'px')
        })
        .on('mouseout', function (d) {
          tooltip.transition()
            .duration(500)
            .style('opacity', 0)
        })
    },
    fillData () {
      if (this.current_grid === '') {
        return
      }
      var self = this
      axios.get('http://127.0.0.1:8000/grids/' + this.current_grid + '/results')
        .then(function (response) {
          self.raw_data = response.data
          self.processData(response.data)
        })
        .catch(function (error) {
          console.log(error)
        })
    }
  }
}
</script>
<style>
  .small {
    max-width: 600px;
    margin:  150px auto;
  }
.v-select .selected-tag {
  color: black;
  background-color: transparent;
  border: none;
  border-radius: 4px;
  height: 26px;
  margin: 4px 1px 0px 3px;
  padding: 1px 0.25em;
  float: left;
  line-height: 24px;
}

.tooltip {
  position: absolute;
}

</style>
