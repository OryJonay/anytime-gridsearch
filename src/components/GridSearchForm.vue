<template>
  <v-layout row wrap>
    <v-flex offset-md3 md5>
      <br>
	  <v-stepper v-model="e6" vertical>
	    <v-stepper-step step="1" v-bind:complete="e6 > 1">
	      Choose a dataset name
	      <small>Dataset names must be unique</small>
	    </v-stepper-step>
	    <v-stepper-content step="1">
	      <v-text-field
	        name="dataset-name"
	        label="Dataset Name"
	        id="dataset-name">
	      </v-text-field>
	      <v-btn class="green" primary light @click.native="e6 = 2">Continue</v-btn>
	      <v-btn light @click.native="e6 = 1">Cancel</v-btn>
	    </v-stepper-content>
	    <v-stepper-step step="2" v-bind:complete="e6 > 2">Upload examples</v-stepper-step>
	    <v-stepper-content step="2">
	      <multiple-file-uploader></multiple-file-uploader>
	      <v-btn class="green" primary light @click.native="e6 = 3">Continue</v-btn>
	      <v-btn light>Cancel</v-btn>
	    </v-stepper-content>
	    <v-stepper-step step="3">Upload labels</v-stepper-step>
	    <v-stepper-content step="3">
	      <multiple-file-uploader></multiple-file-uploader>
	      <v-btn class="green" primary light @click.native="e6 = 4">Continue</v-btn>
	      <v-btn light>Cancel</v-btn>
	    </v-stepper-content>
	    <v-stepper-step step="4">View setup instructions</v-stepper-step>
	    <v-stepper-content step="4">
	      <v-card class="grey lighten-1 z-depth-1 mb-5" height="200px"></v-card>
	      <v-btn router :to="{name: 'GridResult'}" class="green" primary light @click.native="e6 = 1">Continue</v-btn>
	      <v-btn class="green" light>Cancel</v-btn>
	    </v-stepper-content>
	  </v-stepper>
	</v-flex>
  </v-layout>
</template>

<script>
import MultipleFileUploader from 'vue2-multi-uploader'
export default {
  components: {
    MultipleFileUploader
  },
  data () {
    return {
      e6: 1
    }
  }
}
</script>

<style>
.stepper--vertical {
    padding-bottom: 36px;
    max-height: 880px;
}
.stepper__step--active .stepper__step__step {
    background: #4caf50!important;
}
.stepper__step--complete .stepper__step__step {
    background: #4caf50!important;
}
.uploadBox {
    position: relative;
    background: transparent;
    padding: 0 1em 1em;
    margin: 1em;
}
</style>
