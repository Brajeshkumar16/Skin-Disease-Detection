import React, { useState } from 'react';
import './ImageUploader.css'

export const ImageUploader = () => {
  const [selectedImage, setSelectedImage] = useState(null);

  // Function to handle image selection
  // const handleImageUpload = (e) => {
  //   const file = e.target.files[0];
  //   if (file) {
  //     const reader = new FileReader();
  //     reader.onload = (e) => {
  //       setSelectedImage(e.target.result);
  //     };
  //     reader.readAsDataURL(file);
  //   }
  // };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append('image', file);
      console.log(formData)

      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
      };
      reader.readAsDataURL(file);

      try {
        const response = await fetch('http://127.0.0.1:5000/upload-image', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const text = await response.text();
          // setSelectedImage(null); // Clear the selected image
          alert(text); // Show the success message from the backend
        } else {
          alert('Image upload failed');
        }
      } catch (error) {
        console.error('Error uploading image:', error);
      }
    }
  };

  return (
    <div className='container'>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {selectedImage && (
        <div className='image'>
          <h2>Selected Image</h2>
          <img
            src={selectedImage}
            alt="Selected"
          />
        </div>
      )}
    </div>
  );
};
