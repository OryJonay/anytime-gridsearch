<template>
	<div id="vue_base_table" class="center">
	    <table>
	        <thead>
	            <tr>
	                <th v-for="key in columns"
	                    @click="sortBy(key)"
	                    :class="{ active: sortKey == key }">
	                    {{ key | capitalize }}
	                    <span class="arrow" :class="sortOrders[key] > 0 ? 'asc' : 'dsc'">
	                    </span>
	                </th>
	            </tr>
	        </thead>
	        <tbody>
	            <tr v-for="entry in filteredData">
	                <td v-for="key in columns">
	                <router-link :to="{ name: 'GridResult', params: { uuid: entry[key] }}" v-if="key==columns[0]" target="_blank">{{ entry[key] }}</router-link>
			</a>
			<span v-else>
				{{ entry[key] }}
			</span>
	                </td>
	            </tr>
	        </tbody>
	    </table>
	</div>
</template>

<script>

export default {
  name: 'vue_base_table',
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

}
</script>

<style>
body {
  font-family: Helvetica Neue, Arial, sans-serif;
  font-size: 14px;
  color: #444;
}

table {
  border: 2px solid #42b983;
  border-radius: 3px;
  background-color: #fff;
}

th {
  background-color: #42b983;
  color: rgba(255,255,255,0.66);
  cursor: pointer;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

td {
  background-color: #f9f9f9;
}

th, td {
  min-width: 120px;
  padding: 10px 20px;
}

th.active {
  color: #fff;
}

th.active .arrow {
  opacity: 1;
}

.arrow {
  display: inline-block;
  vertical-align: middle;
  width: 0;
  height: 0;
  margin-left: 5px;
  opacity: 0.66;
}

.arrow.asc {
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
  border-bottom: 4px solid #fff;
}

.arrow.dsc {
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
  border-top: 4px solid #fff;
}

.center {
    margin: auto;
    width: 60%;
    padding: 10px;
}
</style>
