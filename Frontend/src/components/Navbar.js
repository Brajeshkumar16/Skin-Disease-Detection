import React from 'react'
import './Navbar.css'
import logo from '../assets/cgu-logo.png'

export const Navbar = () => {
  return (
    <div className='nav'>
        <div className='title'>Skin-Disease</div>
        <div>
            <img src={logo} alt="Skin-disease"/>
        </div>
        <div className='about'>About Us</div>
    </div>
  )
}
