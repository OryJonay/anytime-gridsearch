<template>
  <v-layout row wrap>
    <v-flex offset-md3 md5>
      <br>
      <h6 hidden>{{ current_grid }}</h6>
      <v-select :on-change="updateSelection" :options="selections" :value.sync="selected"></v-select>
      <br>
      <line-chart :chart-data="datacollection" 
      :options="{responsive: false, maintainAspectRatio: true, legend: { display: false }}" 
      :complete_data="complete_data"
      :width="760"
  	  :height="760"></line-chart>
    </v-flex>
  </v-layout>
</template>
<script>
  import LineChart from './LineChart.js'
  import axios from 'axios'
  import vSelect from 'vue-select'

  export default {
    components: {
      LineChart,
      vSelect
    },
    computed: {
      current_grid () {
        var newUuid = this.$store.state.grid
        if (newUuid !== this.uuid) {
          this.uuid = newUuid
          this.selections = []
          this.selected = ''
          this.datacollection = {labels: [], datasets: []}
          this.complete_data = {}
          this.raw_data = []
          this.fillData()
        }
        return this.$store.state.grid
      }
    },
    data () {
      return {
        datacollection: {},
        complete_data: {},
        raw_data: [],
        selected: '',
        selections: [],
        uuid: '',
        yLabel: 'Accuracy'
      }
    },
    methods: {
      updateSelection (val) {
        if (this.selections.length === 0) {
          this.fillData()
          return
        }
        this.selected = val
        var res = this.processData(this.raw_data)
        this.complete_data = res[0]
        this.datacollection = res[1]
      },
      processData (retData) {
        var _labels = {}
        var _datasets = []
        if (this.selections.length === 0) {
          this.selections = Object.keys(retData[0]['params'])
        }
        for (var i = 0; i < retData.length; i++) {
          _labels[retData[i]['params'][this.selected]] = []
        }
        for (i = 0; i < retData.length; i++) {
          _labels[retData[i]['params'][this.selected]]
          .push([retData[i]['params'], retData[i]['score']])
        }
        for (i = 0; i < _labels[Object.keys(_labels)[0]].length; i++) {
          var resData = []
          var resParams = []
          var dataSetColor = '#' + (function co (lor) { return (lor += [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f'][Math.floor(Math.random() * 16)]) && (lor.length === 6) ? lor : co(lor) })('')
          for (var x of Object.keys(_labels)) {
            resData.push(_labels[x][i][1])
            resParams.push(_labels[x][i][0])
          }
          var _name = []
          for (var j = 0; j < this.selections.length; j++) {
            if (this.selections[j] === this.selected) {
              continue
            }
            _name.push([Object.keys(resParams[0])[j], resParams[0][Object.keys(resParams[0])[j]]].join('-'))
          }
          _datasets.push({'label': 'Score ' + _name.join(', '),
            'borderColor': dataSetColor,
            'data': resData,
            'fill': false,
            'backgroundColor': dataSetColor,
            'showLine': false})
        }
        return [_labels, {labels: Object.keys(_labels), datasets: _datasets}]
      },
      fillData () {
        if (this.current_grid === '') {
          return
        }
        var self = this
        axios.get('http://127.0.0.1:8000/grids/' + this.current_grid + '/results')
        .then(function (response) {
          self.raw_data = response.data
          var res = self.processData(response.data)
          self.complete_data = res[0]
          self.datacollection = res[1]
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
</style>
