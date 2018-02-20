<template>
    <div>
    <v-select
    :debounce="250"
    :on-search="getClassifiers"
    :options="classifiers"
    :on-change="updateClassifier"
    placeholder="Choose a classifier"
    label="classifier">
    </v-select>
    <v-select multiple v-model="chosen_params" v-if="clf_params != null"
    :options="clf_params"
    label="name"
    placeholder="Choose hyperparameters for the GridSearch"
    :on-change="updateArg"></v-select>
    <v-list three-line>
      <v-list-tile v-for="item in chosen_params" :key="item.name">
      <popper trigger="click">
        <v-list-tile-content slot="reference">
          <v-list-tile-title v-text="item.name"></v-list-tile-title>
          <v-list-tile-sub-title v-text="item.type"></v-list-tile-sub-title>
          <hyper-parameter-form :item="item"></hyper-parameter-form>
        </v-list-tile-content>
        <span class="popper">{{ prettyDesc(item.desc) }}</span>
      </popper>
      </v-list-tile>
    </v-list>
  <!-- <br v-if="addSpaces" v-for="n in howManySpaces"> -->
  </div>
</template>

<script>
import axios from 'axios'
import vSelect from 'vue-select'
import Popper from 'vue-popperjs'
import 'vue-popperjs/dist/css/vue-popper.css'
import HyperParameterForm from './HyperParameterForm.vue'

export default {
  components: { vSelect, Popper, HyperParameterForm },
  data () {
    return {
      classifiers: [],
      clf: '',
      clf_params: null,
      chosen_params: []
    }
  },
  created () {
    this.getClassifiers('', '')
  },
  computed: {
    addSpaces () {
      return ((this.clf === '') || (this.chosen_params.length !== this.clf_params.length))
    },
    howManySpaces () {
      return (16 - this.chosen_params.length)
    }
  },
  methods: {
    getClassifiers (search, loading) {
      axios.get(`http://127.0.0.1:8000/estimators/`)
        .then(resp => { this.classifiers = resp.data })
        .catch(e => { console.log(e) })
    },
    prettyDesc (desc) {
      return desc.join('\n')
    },
    updateClassifier (val) {
      if (val != null) {
        this.$store.commit('changeClassifier', {'clf': val})
        axios.get(`http://127.0.0.1:8000/estimators/` + val)
          .then(resp => {
            this.clf_params = resp.data
            this.clf = val
            this.chosen_params = []
          })
          .catch(e => { console.log(e) })
      } else {
        this.clf = ''
        this.clf_params = null
        this.chosen_params = []
        this.$store.commit('changeClassifier', {'clf': ''})
      }
    },
    updateArg (val) {
      this.$store.commit('updateArgsObject', {'args': val.slice(0, val.length)})
    }
  }
}
</script>

<style>
.popper {
    width: auto;
    background-color: #fafafa;
    color: #212121;
    text-align: left;
    padding: 2px;
    display: inline-block;
    border-radius: 3px;
    position: relative !important;
    left: 0px !important;
    font-size: 14px;
    font-weight: normal;
    border: 1px #ebebeb solid;
    z-index: 200000;
    -moz-box-shadow: rgb(58, 58, 58) 0 0 6px 0;
    -webkit-box-shadow: rgb(58, 58, 58) 0 0 6px 0;
    box-shadow: rgb(58, 58, 58) 0 0 6px 0;
}
</style>
