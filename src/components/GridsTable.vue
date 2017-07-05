<template>
	<div id="grids_table" class="center">
		<form id="search">
	    	Search <input name="query" v-model="searchQuery">
	  	</form>
		<vue-base-table :data="gridData"
						:columns="gridColumns"
						:filter-key="searchQuery"></vue-base-table>
	</div>
</template>

<script>
	import VueBaseTable from './VueBaseTable.vue'
	import axios from 'axios'
	
	export default {
	  name: 'grids_table',
	  components: {
	    VueBaseTable
	  },
	  data: function () {
	    return {searchQuery: '', gridColumns: ['uuid', 'classifier', 'best_score'], gridData: []}
	  },
	  mounted: function () {
	    this.fetchData()
	  },
	  methods: {
	    fetchData: function () {
      var self = this
      axios.get('http://127.0.0.1:8000/grids/')
      .then(function (response) {
        self.gridData = response.data
        self.gridColumns = Object.keys(response.data[0])
      })
      .catch(function (error) {
        console.log(error)
        self.gridData = []
        self.gridColumns = ['uuid', 'classifier', 'best_score']
      })
    }}
	}
</script>

<style>
.center {
    margin: auto;
    width: 60%;
    padding: 10px;
}
</style>
