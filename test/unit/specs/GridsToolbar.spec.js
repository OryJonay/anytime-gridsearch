import Vue from 'vue'
import GridsToolbar from '@/components/GridsToolbar'

describe('GridsToolbar.vue', () => {
  it('Should render correct toolbar title', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    expect(vm.selected_dataset).to.equal('')
    expect(vm.$el.querySelector('.hidden-sm-and-down').textContent)
      .to.equal('AnyTimeGridSearch github')
  })
  it('Should mount with blank dataset', () => {
    const Constructor = Vue.extend(GridsToolbar)
    const vm = new Constructor().$mount()
    expect(vm.selected_dataset).to.equal('')
  })
})
