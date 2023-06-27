import './assets/css/index.css'
import VFileDropPlugin from 'v-file-drop'
import 'v-file-drop/styles.css'

import { createApp } from 'vue'
import App from './App.vue'

const app = createApp(App)
app.use(VFileDropPlugin, { multiple: false, accept: ['image/jpeg', 'image/png'] })
app.mount('#app')
