import Vuex from 'vuex'
import Vue from 'vue'

Vue.use(Vuex)

var store = new Vuex.Store({
  state: {
    grid: ''
  },
  mutations: {
    change (state, payload) {
      state.grid = payload.uuid
    }
  }
})

export default store
