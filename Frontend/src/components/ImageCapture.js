import React, { useRef, useState } from "react";
import axios from 'axios';
import './ImageUploader.css'

export const ImageCapture = () => {
  const videoRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [stream, setStream] = useState(null);
  const [result, setResult] = useState("")

  // Function to start capturing
  const startCapture = async () => {
    setCapturedImage(null);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(mediaStream);
      videoRef.current.srcObject = mediaStream;
      setIsCapturing(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  // Function to stop capturing
  const stopCapture = () => {
    if (stream) {
      const tracks = stream.getTracks();
      tracks.forEach((track) => {
        track.stop();
      });
    }

    setIsCapturing(false);
    setStream(null);
  };

  // Function to take a snapshot
  // const takeSnapshot = () => {
  //   const canvas = document.createElement('canvas');
  //   const video = videoRef.current;
  //   canvas.width = video.videoWidth;
  //   canvas.height = video.videoHeight;
  //   const ctx = canvas.getContext('2d');
  //   ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  //   const imageUrl = canvas.toDataURL('image/png');
  //   setCapturedImage(imageUrl);
  //   stopCapture()
  // };


  const takeSnapshotAndUpload = async () => {
    const canvas = document.createElement('canvas');
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageUrl = canvas.toDataURL('image/png');
    setCapturedImage(imageUrl);
    

    try{
      const imageBlob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));

      const formData = new FormData();
      formData.append('image', imageBlob);

      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log(response.data.message);
      setResult(response.data.message)
    }catch(error){
      console.error('Error uploading image:', error);
    }
    stopCapture()
  };

  return (
    <div>
      <button className="scanBtn" onClick={isCapturing ? stopCapture : startCapture}>
        {isCapturing ? 'Stop Capture' : 'Start Scanning'}
      </button>
      {isCapturing && (
        <button onClick={takeSnapshotAndUpload} disabled={!isCapturing}>
          Take Picture
        </button>
      )}
      {result && (
        <div>{result}</div>
      )}
      {capturedImage && (
        <div className="captured">
          {/* <button onClick={takeSnapshotAndUpload}>Send Image to Backend</button> */}
          <h2>Captured Image</h2>
          <img src={capturedImage} alt="Captured" />
        </div>
      )}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: isCapturing ? 'inline' : 'none' }}
      />
    </div>
  );
};
