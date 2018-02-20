<template>
  <span id="app">
    <grids-toolbar></grids-toolbar>
    <router-view></router-view>
  </span>
</template>

<script>
import 'vuetify/dist/vuetify.min.css'
import store from './store.js'
import GridsToolbar from './components/GridsToolbar.vue'
export default {
  name: 'app',
  store,
  components: {
    GridsToolbar
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
