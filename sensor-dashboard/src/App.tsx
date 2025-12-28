import { useState, useEffect } from 'react'
import SensorDisplay from './components/SensorDisplay'

  const SENSOR_NAMES = {
    s_1: "Fan Inlet Temp",
    s_2: "LPC Outlet Temp",
    s_3: "HPC Outlet Temp",
    s_4: "LPT Outlet Temp",
    s_5: "Fan Inlet Pressure",
    s_6: "Bypass Duct Press",
    s_7: "HPC Outlet Press",
    s_8: "Phys Fan Speed",
    s_9: "Phys Core Speed",
    s_10: "Engine Press Ratio",
    s_11: "Static HPC Outlet P",
    s_12: "Fuel Flow Ratio",
    s_13: "Corr Fan Speed",
    s_14: "Corr Core Speed",
    s_15: "Bypass Ratio",
    s_16: "Burner Burner Ratio",
    s_17: "Bleed Enthalpy",
    s_18: "Demanded Fan Speed",
    s_19: "Demanded Corr Fan Speed",
    s_20: "HPT Coolant Bleed",
    s_21: "LPT Coolant Bleed"
  };

  const SETTING_NAMES = {
    setting_1: "Altitude",
    setting_2: "Mach Number",
    setting_3: "Throttle Resolver Angle"
  };

  const ALL_SENSOR_NAMES = { ...SENSOR_NAMES, ...SETTING_NAMES };

function App() {

  const [windowRange, setWindowRange] = useState({ startIndex: 0, endIndex: 50 });

  const handleBrushChange = (newRange) => {
      // newRange looks like: { startIndex: 10, endIndex: 60 }
      // ... logic goes here ...
      setWindowRange(newRange);
  }

  const fetchData = async () => {
    try {
      console.log("Fetching data from Python...");
      
      // 1. Make the request
      //const response = await fetch("http://127.0.0.1:8001/sensors/?limit=1");
      // Inside fetchData
      const response = await fetch(
        `http://127.0.0.1:8001/sensors/?start_cycle=${windowRange.startIndex}&end_cycle=${windowRange.endIndex}&limit=500`
      );

      const data = await response.json();




      const latestReading = data[0]; 
      console.log("Latest Reading:", latestReading);

      const formattedSensors = Object.keys(ALL_SENSOR_NAMES).map((key, index) => {
        return {
          id: index,
          label: ALL_SENSOR_NAMES[key],
          history: data.map(row => ({value: row[key] })),
        };
      })

     // const formattedSensors = Object.keys(latestReading)
       //       .filter((key) => key.startsWith('s_')) // Only keep sensor keys
         //     .map((key, index) => ({
           //     id: index,
            //    label: SENSOR_NAMES[key] || key, // Use the nice name, or fallback to 's_1'
             //   value: latestReading[key] // Get the number (e.g., 518.67)
             // }));

      setSensors(formattedSensors);
      //setCount(data.predicted_rul);
      
    } catch (error) {
      console.error("Error fetching data:", error);
    };
  };

  useEffect(() => {
      // Call the function immediately
      fetchData();
    }, [windowRange]);

  const [sensors, setSensors] = useState([
    { id: 1, label: 'Loading...', value: 0, history: [] }
  ]);

  return (
    <>
      <h1>Engine Dashboard</h1>
      <div className="card">
        <button onClick={() => fetchData()}>
          Update Sensor
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      {/* 2. Map over the 'sensors' state variable now */}
      {sensors.map((sensor) => (
        <SensorDisplay 
          key={sensor.id}
          label={sensor.label}
          
          // Pass the full array for the chart
          history={sensor.history} 
          
          // Pass ONLY the last number for the big text display
          // We check if history exists to avoid errors during loading
          value={sensor.history.length > 0 ? sensor.history[sensor.history.length - 1].value : 0} 
        />
      ))}
    </>
  )
}

export default App

