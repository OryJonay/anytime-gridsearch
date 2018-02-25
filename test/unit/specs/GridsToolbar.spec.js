import Vue from 'vue'
import moxios from 'moxios'
import GridsToolbar from '@/components/GridsToolbar'

describe('GridsToolbar.vue', () => {
  beforeEach(function () {
    moxios.install()
  })
  afterEach(function () {
    moxios.uninstall()
  })
  it('Should render correct toolbar title', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    expect(vm.$el.querySelector('.hidden-sm-and-down').textContent)
      .to.equal('AnyTimeGridSearch github')
  })
  it('Should mount with blank dataset', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    expect(vm.selected_dataset).to.equal('')
  })
  it('Should mount and get a list of datasets', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    moxios.stubRequest('/datasets/', {
      status: 200,
      response: [{'name': 'IRIS', 'examples': 'http://127.0.0.1:8000/datasets/datasets/IRIS/examples.csv', 'labels': 'http://127.0.0.1:8000/datasets/datasets/IRIS/labels.csv'}]
    })
    moxios.wait(function () {
      expect(vm.data().datasets.length).to.equal(1)
    })
  })
})
