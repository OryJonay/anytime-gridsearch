Vue.component('line-chart',{
  extends: VueChartJs.Line,
  delimiters : ["[[","]]"],
  props: {labels: Array, datasets: Array},
  mounted : function(){
    this.renderChart({labels:this.labels,datasets:this.datasets}, {responsive: true, maintainAspectRatio: false})
  },
//  methods: {
//	  fetchData: function(){
//		  var xhr = new XMLHttpRequest()
//	      var self = this
//	      xhr.open('GET', 'http://127.0.0.1:8000/grids/1be28f9e-fd71-496f-8b05-c8beeea37c4c/results')
//	      xhr.onload = function () {
//	    	  	ret_data = JSON.parse(xhr.responseText)
//	    	  	
//		        for (i=1; i<=ret_data.length; i++) {
//		            self.labels.push(i)
//		        }
//	    	  	self.datasets = []
//	    	  	res_data = []
//	    	  	for (i=1; i<=ret_data.length; i++) {
//		            res_data.push(ret_data[i-1]['score'])
//		        }
//	    	  	self.datasets.push({'label':'MLP','backgroundColor':'#f87979',
//	    	  		'data':res_data})
//	      	}
//	      xhr.send()
//	  }
//  }

});

var vm = new Vue({
  el: '.app',
  delimiters : ["[[","]]"],
  data: {
    message: 'Hello World',
    labels: [],
    datasets: [],
  },
  
  mounted: function () {
	  	this.fetchData()
	  },
  methods: {
	  fetchData: function(){
		  var xhr = new XMLHttpRequest()
	      var self = this
	      xhr.open('GET', 'http://127.0.0.1:8000/grids/1be28f9e-fd71-496f-8b05-c8beeea37c4c/results')
	      xhr.onload = function () {
	    	  	ret_data = JSON.parse(xhr.responseText)
		        for (i=1; i<=ret_data.length; i++) {
		            self.labels.push(i)
		        }
	    	  	self.datasets = []
	    	  	res_data = []
	    	  	for (i=1; i<=ret_data.length; i++) {
		            res_data.push(ret_data[i-1]['score'])
		        }
	    	  	self.datasets.push({'label':'MLP','backgroundColor':'#f87979',
	    	  		'data':res_data})
	      	}
	      xhr.send()
	  }
	  }
})


Vue.component('demo-grid', {
  template: '#grid-template',
  delimiters: ['[[',']]'],
  props: {
    data: Array,
    columns: Array,
    filterKey: String
  },
  data: function () {
    var sortOrders = {}
    this.columns.forEach(function (key) {
      sortOrders[key] = 1
    })
    return {
      sortKey: '',
      sortOrders: sortOrders
    }
  },
  computed: {
    filteredData: function () {
      var sortKey = this.sortKey
      var filterKey = this.filterKey && this.filterKey.toLowerCase()
      var order = this.sortOrders[sortKey] || 1
      var data = this.data
      if (filterKey) {
        data = data.filter(function (row) {
          return Object.keys(row).some(function (key) {
            return String(row[key]).toLowerCase().indexOf(filterKey) > -1
          })
        })
      }
      if (sortKey) {
        data = data.slice().sort(function (a, b) {
          a = a[sortKey]
          b = b[sortKey]
          return (a === b ? 0 : a > b ? 1 : -1) * order
        })
      }
      return data
    }
  },
  filters: {
    capitalize: function (str) {
      return str.charAt(0).toUpperCase() + str.slice(1)
    }
  },
  methods: {
    sortBy: function (key) {
      this.sortKey = key
      this.sortOrders[key] = this.sortOrders[key] * -1
    }
  }
})

var demo = new Vue({
  el: '#demo',
  delimiters: ['[[',']]'],
  data: {
    searchQuery: '',
    gridColumns: ['best_score'],
    gridData: []
  },
  created: function () {
  	this.fetchData()
  },
  methods: {
      fetchData: function () {
          var xhr = new XMLHttpRequest()
          var self = this
          xhr.open('GET', 'http://127.0.0.1:8000/grids/')
          xhr.onload = function () {
		        self.gridData = JSON.parse(xhr.responseText)
		        self.gridColumns = Object.keys(JSON.parse(xhr.responseText)[0]) 
          	}
          xhr.send()
      	}}
})
