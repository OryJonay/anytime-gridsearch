<template>
  <v-toolbar class="green" light>
      <v-toolbar-items>
        <v-toolbar-item>
          <v-select
          :options="dataset_names"
          :on-change="updateDataset"
          placeholder="Choose a dataset"
          label="name"
          class="datasets-select"></v-select>
        </v-toolbar-item>
        <v-toolbar-item>
          <v-btn icon light class="pl-3">Settings<v-icon fa light class="pl-1">cogs</v-icon></v-btn>
        </v-toolbar-item>
      </v-toolbar-items>
      <v-toolbar-title class="hidden-sm-and-down">AnyTimeGridSearch <v-icon fa light>github</v-icon></v-toolbar-title>
      <v-toolbar-items>
        <v-toolbar-item>
          <v-btn router icon light :to="{name:'GridSearchForm'}">New<v-icon>add</v-icon></v-btn>
        </v-toolbar-item>
        <v-toolbar-item>
          <v-select
          :options="classifiers"
          :on-change="updateGrid"
          placeholder="Choose a classifier"
          label="classifier"
          class="classifiers-select"></v-select>
        </v-toolbar-item>
      </v-toolbar-items>
  </v-toolbar>
</template>

<script>
import axios from 'axios'
import vSelect from 'vue-select'

export default {
  components: { vSelect },
  data () {
    return {
      dataset_names: [],
      classifiers: [],
      selected_dataset: ''
    }
  },
  created () {
    this.getDatasets('', '')
  },
  methods: {
    getDatasets (search, loading) {
      axios.get(`/datasets/`)
        .then(resp => { this.dataset_names = resp.data })
        .catch(e => { console.log(e) })
    },
    updateDataset (val) {
      this.selected_dataset = val.name
      this.getClassifiers()
    },
    updateGrid (val) {
      this.$store.commit('changeGrid', {'uuid': val.uuid})
    },
    getClassifiers () {
      if (this.selected_dataset === '') { return }
      axios.get(`/datasets/` + this.selected_dataset + `/grids`)
        .then(resp => { this.classifiers = resp.data })
        .catch(e => { console.log(e) })
    }
  }
}
</script>

<style>
.datasets-select {
  width: 410px;
}
.classifiers-select {
  width: 410px;
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
.v-select .dropdown-toggle .clear {
    position: absolute;
    bottom: 9px;
    right: 30px;
    font-size: 23px;
    font-weight: 700;
    line-height: 1;
    color: rgba(60,60,60,.5);
    padding: 0;
    border: 0;
    background-color: transparent;
    cursor: pointer;
    visibility: hidden;
}
</style>
