// "use client"
// import {useState} from "react"
// import React from 'react'

// const Form = () => {

//     const [hexData, setHexData] = useState('');
//     const [prediction, setPrediction] = useState('');


//   return (
//     <div>  <div className="wrapper ml-5 mr-5 w-[70vw]  bg-white rounded-sm shadow-lg">

//     <div className="card px-8 py-4">
//         <div className="card-image mt-10 mb-6">
//             <img src="./treeslogo.png" className='h-[5rem]'></img>
//         </div>

//         <div className="card-text">
//             <h1 className="text-xl md:text-2xl font-bold leading-tight text-gray-900">Random Forest Classifier</h1>
//             <p className="text-base md:text-lg text-gray-700 mt-3 ">Harness the power of a Random Forest Classifier to accurately identify encryption algorithms, transforming encrypted data analysis into a breeze!</p>
//         </div>

//         <div className="card-mail flex items-center my-10">
//             <input type="text" className="border-l border-t border-b border-gray-200 rounded-l-md w-full text-base md:text-lg px-3 py-2" placeholder="Enter Your Hexadecimal Cipher"></input>
//             <button className="bg-orange-500 hover:bg-orange-600 hover:border-orange-600 text-white font-bold capitalize px-3 py-2 text-base md:text-lg rounded-r-md border-t border-r border-b border-orange-500">Predict</button>
//         </div>
//     </div>
// </div>
// </div>
//   )
// }

// export default Form


// "use client"
// import React, { useState } from 'react';
// import axios from 'axios';

// const Form = () => {
//   const [hexData, setHexData] = useState('');
//   const [prediction, setPrediction] = useState('');
//   const [error, setError] = useState('');

//   const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
//     e.preventDefault();
//     setError(''); 
    
//     if (!hexData) {
//       setError('Please provide hexadecimal data.');
//       return;
//     }

//     try {
     
//       const response = await axios.post('http://localhost:5002/predict', { hexa: hexData });

//       if (response.data.error) {
//         console.log(response.data.error)
//         setError(response.data.error);
//       } else {
//         setPrediction(response.data.prediction || 'Prediction unavailable');
//         console.log(response.data.prediction)
//       }
//     } catch (err) {
//       setError('An error occurred while predicting. Please try again later.');
//       console.error(err);
//     }
//   };

//   return (
//     <div className="wrapper ml-5 mr-5 w-[70vw] bg-white rounded-sm shadow-lg">
//       <div className="card px-8 py-4">
//         <div className="card-image mt-10 mb-6">
//           <img src="./treeslogo.png" className="h-[5rem]" alt="Logo" />
//         </div>

//         <div className="card-text">
//           <h1 className="text-xl md:text-2xl font-bold leading-tight text-gray-900">Random Forest Classifier</h1>
//           <p className="text-base md:text-lg text-gray-700 mt-3">
//             Harness the power of a Random Forest Classifier to accurately identify encryption algorithms, transforming encrypted data analysis into a breeze!
//           </p>
//         </div>

//         <form onSubmit={handleSubmit}>
//           <div className="card-mail flex items-center my-10">
//             <input
//               type="text"
//               className="border-l border-t border-b border-gray-200 rounded-l-md w-full text-base md:text-lg px-3 py-2"
//               placeholder="Enter Your Hexadecimal Cipher"
//               value={hexData}
//               onChange={(e) => setHexData(e.target.value)}
//             />
//             <button
//               type="submit"
//               className="bg-orange-500 hover:bg-orange-600 hover:border-orange-600 text-white font-bold capitalize px-3 py-2 text-base md:text-lg rounded-r-md border-t border-r border-b border-orange-500"
//             >
//               Predict
//             </button>
//           </div>
//         </form>

//         {error && <p className="text-red-500 text-sm">{error}</p>}

//         {prediction && (
//           <div className="mt-4 p-4 bg-green-100 border-l-4 border-green-500 text-green-700">
//             <p><strong>Prediction:</strong> {prediction}</p>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default Form;

"use client"
import { useState } from 'react';
import axios from 'axios';

export default function Form() {
  const [hexInput, setHexInput] = useState('');
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setPrediction('');
    setError('');

    try {
      const formData = new FormData();
      formData.append('hexa', hexInput);

      const response = await axios.post('http://127.0.0.1:5002/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data) {
        setPrediction(`The algorithm is ${response.data[0]}`);
        console.log(response.data[0]);
        setHexInput('');
      }
      else {
        setError('Unexpected response format from server.');
        console.log(response.data);
        console.log(response.data[0]);
      }
    } catch (err) {
      console.error(err);
      setError('Something went wrong. Please try again.');
    }
  };

  return (
    <div className="container">
      <h1>Hexadecimal Predictor</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={5}
          placeholder="Enter hexadecimal data"
          value={hexInput}
          onChange={(e) => setHexInput(e.target.value)}
          required
        ></textarea>
        <button type="submit">Predict</button>
      </form>
      {prediction && <p className="result">{prediction}</p>}
      {error && <p className="error">{error}</p>}
      <style jsx>{`
        .container {
          width: 60%;
          margin: 0 auto;
          text-align: center;
        }
        textarea {
          width: 100%;
          padding: 10px;
          margin: 10px 0;
        }
        button {
          padding: 10px 20px;
          background-color: #0070f3;
          color: white;
          border: none;
          cursor: pointer;
        }
        .result {
          color: green;
          margin-top: 10px;
        }
        .error {
          color: red;
          margin-top: 10px;
        }
      `}</style>
    </div>
  );
}
