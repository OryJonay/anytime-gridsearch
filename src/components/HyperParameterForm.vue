<template>
  <div>
    <div v-if="item.type.indexOf('int') > -1 || item.type.indexOf('float') > -1">
      <input v-model.number="start" type="number" placeholder="Start" @input="updateValue($event.target)" :name="item.name">
      <input v-model.number="end" type="number" placeholder="End" @input="updateValue($event.target)" :name="item.name">
      <input v-model.number="skip" type="number" placeholder="Skip" @input="updateValue($event.target)" :name="item.name">
    </div>
    <div v-if="item.type.indexOf('bool') > -1">
      <input type="checkbox" id="true" placeholder="boolean" value="True" v-model="checkedOptions" @change="updateBool($event.target)" :name="item.name">
      <label for="true">True</label>
      <input type="checkbox" id="false" placeholder="boolean" value="False" v-model="checkedOptions" @change="updateBool($event.target)" :name="item.name">
      <label for="false">False</label>
    </div>
    <div v-if="item.type.indexOf('string') > -1">
      <input v-model="stringOptions" placeholder="Write different options separated by a comma (,)" style="width: 1000px" @input="updateValue($event.target)" :name="item.name">
    </div>
  </div>
</template>

<script>
export default {
  data () {
    return {
      start: null,
      end: null,
      skip: null,
      checkedOptions: [],
      stringOptions: ''
    }
  },
  props: { 'item': Object },
  methods: {
    updateValue (val) {
      this.$store.commit('addArgument', {'name': val.name, 'type': val.placeholder.toLowerCase(), 'value': val.value})
    },
    updateBool (val) {
      this.$store.commit('addArgument', {'name': val.name, 'type': val.placeholder.toLowerCase(), 'value': this.checkedOptions})
    }
  }
}
</script>

<style>
</style>
