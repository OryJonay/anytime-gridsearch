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
  it('Should have 2 toolbar items items', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    expect(vm.$el.querySelectorAll('v-toolbar-items').length)
      .to.equal(2)
  })
  it('Should have 4 toolbar item items', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    expect(vm.$el.querySelectorAll('v-toolbar-item').length)
      .to.equal(4)
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
  it('Should mount with blank classifiers', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    expect(vm.classifiers.length).to.equal(0)
  })
  it('Should mount with blank dataset names', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    expect(vm.dataset_names.length).to.equal(0)
  })
  it('Should mount and get a list of datasets', (done) => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    moxios.wait(function () {
      let request = moxios.requests.mostRecent()
      if (request.url === '/datasets/') {
        request.respondWith({
          status: 200,
          response: [{'name': 'IRIS', 'examples': 'http://127.0.0.1:8000/datasets/datasets/IRIS/examples.csv', 'labels': 'http://127.0.0.1:8000/datasets/datasets/IRIS/labels.csv'}]
        }).then(function () {
          expect(vm.dataset_names.length).to.equal(1)
          done()
        }).catch(done)
      }
    })
  })
})
