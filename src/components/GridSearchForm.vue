<template>
  <v-layout row wrap>
    <v-flex offset-md1 md10>
      <br>
	  <v-stepper v-model="e6" vertical  overflow="scroll">
	    <v-stepper-step step="1" v-bind:complete="e6 > 1">
	      Create a dataset
	      <small>Dataset names must be unique</small>
	    </v-stepper-step>
	    <v-stepper-content step="1">
	      <v-text-field
	        v-model="datasetName"
	        name="dataset-name"
	        label="Dataset Name"
	        id="dataset-name"
	        required>
	      </v-text-field>
	      <dropzone 
	        id="dataSetsDropzone"
	        url="http://127.0.0.1:8000/datasets/"
	        :show-remove-link="true"
	        :max-file-size-in-m-b="25"
	        :preview-template="previewTemplate"
	        :dropzone-options="options"
	        :timeout="30000000"
	        :use-custom-dropzone-options="true"><input name="dataset" type="hidden" :value="datasetName"/></dropzone>
	      <v-btn v-if="datasetName != ''" class="green" primary light @click.native="e6 = 2">Continue</v-btn>
	      <v-btn light @click.native="e6 = 1">Cancel</v-btn>
	    </v-stepper-content>
	    <v-stepper-step step="2" v-bind:complete="e6 > 2">Setup Classifier Grid</v-stepper-step>
	    <v-stepper-content step="2" style="{height: 500px}">
	      <classifier-grid-form></classifier-grid-form>
	      <v-btn v-if="current_clf != '' && current_clf != null" router :to="{name: 'GridResult'}" 
	      class="green" primary light @click.native="finishClassifier">Continue</v-btn>
	      <v-btn light @click.native="e6 = 1">Cancel</v-btn>
	    </v-stepper-content>
	  </v-stepper>
	</v-flex>
  </v-layout>
</template>

<script>

import Dropzone from 'vue2-dropzone'
import ClassifierGridForm from './ClassifierGridForm.vue'
import axios from 'axios'

export default {
  components: {
    Dropzone,
    ClassifierGridForm
  },
  data () {
    return {
      e6: 1,
      datasetName: '',
      options: {
        'uploadMultiple': true
      }
    }
  },
  computed: {
    current_clf () {
      return this.$store.state.clf
    }
  },
  methods: {
    previewTemplate () {
      return '<div class="dz-preview dz-file-preview">' +
      '<div class="dz-image" style="width: 150px;height: 150px">' +
      '<img data-dz-thumbnail /></div>' +
      '<div class="dz-details">' +
      '<div class="dz-size"><span data-dz-size></span></div>' +
      '<div class="dz-filename"><span data-dz-name></span></div></div>' +
      '<div class="dz-progress"><span class="dz-upload" data-dz-uploadprogress></span></div>' +
      '<div class="dz-error-message"><span data-dz-errormessage></span></div></div>'
    },
    finishClassifier () {
      var data = { 'clf': this.$store.state.clf, 'dataset': this.datasetName }
      data['args'] = this.$store.state.args
      delete data['args']['__ob__']
      axios.post(`http://127.0.0.1:8000/gridsearch_create/`, data)
      .then(resp => {
        var uuid = resp.data
        console.log(uuid)
      })
      .catch(e => { console.log(e) })
      this.$store.commit('clearCLFForm')
    }
  }
}
</script>

<style>
.stepper--vertical {
  padding-bottom: 36px;
}
.stepper__step--active .stepper__step__step {
  background: #4caf50!important;
}
.stepper__step--complete .stepper__step__step {
  background: #4caf50!important;
}
</style>
