import Vuex from 'vuex'
import Vue from 'vue'

Vue.use(Vuex)

Set.prototype.difference = function (setB) {
  var difference = new Set(this)
  for (var elem of setB) {
    difference.delete(elem)
  }
  return difference
}

var store = new Vuex.Store({
  state: {
    grid: '',
    clf: '',
    args: {}
  },
  mutations: {
    changeGrid (state, payload) {
      state.grid = payload.uuid
    },
    changeClassifier (state, payload) {
      state.clf = payload.clf
    },
    addArgument (state, payload) {
      if (state.args[payload.name] === undefined) {
        state.args[payload.name] = {}
      }
      state.args[payload.name][payload.type] = payload.value
    },
    updateArgsObject (state, payload) {
      var exisiting = new Set(Object.keys(state.args))
      var adding = new Set(payload.args.map(function (v) { return v.name }))
      var toRemove = exisiting.difference(adding)
      var toAdd = adding.difference(exisiting)
      for (let elm of toRemove) {
        delete state.args[elm]
      }
      for (let elem of toAdd) {
        state.args[elem] = {}
      }
    },
    clearCLFForm (state) {
      state.clf = ''
      state.args = {}
    }
  }
})

export default store
