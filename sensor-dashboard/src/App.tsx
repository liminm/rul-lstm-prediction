import { useState } from 'react'
import SensorDisplay from './components/SensorDisplay'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <h1>Engine Dashboard</h1>
      <div className="card">
        <SensorDisplay value={count} label="Sensor Count" />
        <button onClick={() => setCount((count) => count + 1)}>
          Update Sensor
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
    </>
  )
}

export default App
