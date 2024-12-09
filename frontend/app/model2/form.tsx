"use client";
import { useState } from "react";
import React from "react";
import axios from "axios";

export default function Form() {
  const [hexData, setHexData] = useState("");
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState("");

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    setPrediction("");

    try {
      const formdata = new FormData();
      formdata.append("hexa", hexData);

      const response = await axios.post(
        "http://127.0.0.1:5003/predict",
        formdata,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      if (response.data) {
        setPrediction(`The algorithm is ${response.data[0]}`);
        console.log(response.data[0]);
        setHexData("");
      } else {
        setError("Unexpected response format from server.");
        console.log(response.data);
        setHexData("");
      }
    } catch (err) {
        setHexData("");
      console.error(err);
      setError("Wrong Input Format. Please enter a hexadecimal string.");
      
    }
  };

  return (
    <div className="wrapper ml-5 mr-5 w-[70vw] bg-white rounded-sm shadow-lg">
      <div className="card px-8 py-4">
        <div className="card-image mt-10 mb-6">
          <img src="./treeslogo.png" className="h-[5rem]" alt="Logo" />
        </div>

        <div className="card-text">
          <h1 className="text-xl md:text-2xl font-bold leading-tight text-gray-900">
            XGBoost Classifier
          </h1>
          <p className="text-base md:text-lg text-gray-700 mt-3">
            Harness the power of a XGBoost Classifier to accurately
            identify encryption algorithms, transforming encrypted data analysis
            into a breeze!
          </p>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="card-mail flex items-center my-10">
            <input
              type="text"
              className="border-l border-t border-b border-gray-200 rounded-l-md w-full text-base md:text-lg px-3 py-2"
              placeholder="Enter Your Hexadecimal Cipher"
              value={hexData}
              onChange={(e) => setHexData(e.target.value)}
            />
            <button
              type="submit"
              className="bg-orange-500 hover:bg-orange-600 hover:border-orange-600 text-white font-bold capitalize px-3 py-2 text-base md:text-lg rounded-r-md border-t border-r border-b border-orange-500"
            >
              Predict
            </button>
          </div>
        </form>

        {error && <p className="text-red-500 text-sm">{error}</p>}

        {prediction && (
          <div className="mt-4 p-4 bg-green-100 border-l-4 border-green-500 text-green-700">
            <p>
              <strong>Prediction:</strong> {prediction}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
