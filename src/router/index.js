import Vue from 'vue'
import Router from 'vue-router'
import CVResultChart from '@/components/CVResultChart'
import GridSearchForm from '@/components/GridSearchForm'
// import App from '@/App'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'GridResult',
      component: CVResultChart
    },
    {
      path: '/new',
      name: 'GridSearchForm',
      component: GridSearchForm
    }
  ]
})
