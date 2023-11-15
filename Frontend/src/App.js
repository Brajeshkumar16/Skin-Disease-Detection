import React, { useEffect, useState } from "react";
import axios from 'axios';
import { ImageUploader } from "./components/ImageUploader";
import './App.css'
import { Navbar } from "./components/Navbar";
import { ImageCapture } from "./components/ImageCapture";

function App() {
  const [data, setData] = useState("");
  const [year, setYear] = useState('');
  const [isLeapYear, setIsLeapYear] = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/data")
      .then((response) => response.json())
      .then((data) => {
        setData(data.message);
        console.log(data)
      })

      .catch((error) => console.error("Error:", error));
  }, []);

  const checkLeapYear = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/leap', { year });

      if (response.status === 200) {
        const data = response.data;
        setIsLeapYear(data.isLeapYear);
      } else {
        setIsLeapYear(null);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <Navbar/>
      <h1>Skin Disease Detector</h1>
      {/* <input
        type="number"
        placeholder="Enter a year"
        value={year}
        onChange={(e) => setYear(e.target.value)}
      />
      <button onClick={checkLeapYear}>Upload</button> */}
      {/* {isLeapYear !== null? (
        <p>
          {year} is {isLeapYear ? 'a leap year.' : 'not a leap year.'}
        </p>
      ):null} */}
      <div>
        <ImageUploader/>
      </div>

      <div>
        <ImageCapture/>
      </div>
    </div>
  );
}

export default App;
