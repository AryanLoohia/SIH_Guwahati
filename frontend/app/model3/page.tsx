import React from 'react';
import Form from './form';

const Page = () => {
  return (
    <div
      className="h-screen bg-cover bg-center"
      style={{ backgroundImage: `url('/forest.jpg')` }}
    >
      <div className="bg-black bg-opacity-50 h-full flex items-center justify-center">
        <Form />
      </div>
    </div>
  );
};

export default Page;
