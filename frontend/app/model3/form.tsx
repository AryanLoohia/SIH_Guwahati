import React from 'react'

const Form = () => {
  return (
    <div>  <div className="wrapper ml-5 mr-5 w-[70vw]  bg-white rounded-sm shadow-lg">

    <div className="card px-8 py-4">
        <div className="card-image mt-10 mb-6">
            <img src="./neuralnetwork.png" className='h-[5rem]'></img>
        </div>

        <div className="card-text">
            <h1 className="text-xl md:text-2xl font-bold leading-tight text-gray-900">Neural Network</h1>
            <p className="text-base md:text-lg text-gray-700 mt-3 ">Unlock the secrets of encryption with the precision of a Neural Network—decoding algorithmic patterns and revolutionizing cryptographic identification like never before!</p>
        </div>

        <div className="card-mail flex items-center my-10">
            <input type="text" className="border-l border-t border-b border-gray-200 rounded-l-md w-full text-base md:text-lg px-3 py-2" placeholder="Enter Your Hexadecimal Cipher"></input>
            <button className="bg-blue-500 hover:bg-blue-600 hover:border-blue-600 text-white font-bold capitalize px-3 py-2 text-base md:text-lg rounded-r-md border-t border-r border-b border-blue-500">Predict</button>
        </div>
    </div>
</div>
</div>
  )
}

export default Form