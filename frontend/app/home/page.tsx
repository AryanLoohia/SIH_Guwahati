import React from 'react';
import { Link } from 'react-router-dom'; // Import Link for internal navigation
import { AnimatedPinDemo } from './card';
import GoogleGeminiEffectDemo from './bg';

const page = () => {
  // Example prop values for three different sets
  const set1 = {
    title: "Predict",
    linkhref: "./model1", // Change to internal route
    modelname: "Random Forest",
    description: "Harness the power of a Random Forest Classifier to accurately identify encryption algorithms, transforming encrypted data analysis into a breeze!.",
    bgimg:"./forest.jpg",
  };

  const set2 = {
    title: "Predict",
    linkhref: "./model2", // Change to internal route
    modelname: "XGBoost",
    description: "Unlock the secrets of encryption with the precision of a XGBoost.",
    bgimg:"./neuralbackground.jpg"
  };

  

  return (
    <div className='min-h-screen flex flex-col bg-black'>
      <div className="">
        <GoogleGeminiEffectDemo />
      </div>

      <div className="flex mt-0 flex-row bg-black relative -top-[10vh]">
        <AnimatedPinDemo
          title={set1.title}
          linkhref={set1.linkhref}  // Use Link component for internal navigation
          modelname={set1.modelname}
          description={set1.description}
          bgimg={set1.bgimg}
        />
        <AnimatedPinDemo
          title={set2.title}
          linkhref={set2.linkhref} // Use Link component for internal navigation
          modelname={set2.modelname}
          description={set2.description}
          bgimg={set2.bgimg}
        />
        
      </div>
    </div>
  );
};

export default page;
