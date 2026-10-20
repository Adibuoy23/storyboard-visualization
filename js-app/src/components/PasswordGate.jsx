import { useState } from 'react';

// Set your password and Google Form URL in src/appConfig.js
import { APP_PASSWORD, REQUEST_FORM_URL } from '../appConfig.js';

export default function PasswordGate({ onUnlock }) {
  const [input, setInput] = useState('');
  const [error, setError] = useState('');

  function handleSubmit(e) {
    e.preventDefault();
    if (input === APP_PASSWORD) {
      onUnlock();
    } else {
      setError('Incorrect password.');
      setInput('');
    }
  }

  const inputStyle = {
    width: '100%', padding: '10px 12px', fontSize: 14,
    border: '1px solid #ddd', borderRadius: 6,
    boxSizing: 'border-box', fontFamily: 'Arial, sans-serif', outline: 'none',
  };

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      minHeight: '100vh', background: '#f4f5f7', fontFamily: 'Arial, sans-serif',
    }}>
      <div style={{
        background: 'white', borderRadius: 12, padding: '40px 36px',
        boxShadow: '0 2px 16px rgba(0,0,0,0.10)', width: 380, maxWidth: '90vw',
      }}>
        <h2 style={{ margin: '0 0 4px', color: '#2c3e50', fontSize: 20, fontWeight: 700 }}>
          KM and EB Visualization
        </h2>
        <p style={{ margin: '0 0 28px', color: '#888', fontSize: 13 }}>
          Supplementary material — restricted access
        </p>

        <form onSubmit={handleSubmit}>
          <label style={{ fontSize: 12, fontWeight: 600, color: '#555', display: 'block', marginBottom: 6 }}>
            Password
          </label>
          <input
            type="password"
            value={input}
            onChange={e => { setInput(e.target.value); setError(''); }}
            placeholder="Enter password"
            style={{ ...inputStyle, marginBottom: error ? 8 : 14 }}
            autoFocus
          />
          {error && (
            <p style={{ color: '#c0392b', fontSize: 12, margin: '0 0 10px' }}>{error}</p>
          )}
          <button type="submit" style={{
            width: '100%', padding: 11, fontSize: 14, fontWeight: 600,
            background: '#2c3e50', color: 'white', border: 'none',
            borderRadius: 6, cursor: 'pointer', marginBottom: 16,
          }}>
            Enter
          </button>
        </form>

        <div style={{ textAlign: 'center', borderTop: '1px solid #f0f0f0', paddingTop: 16 }}>
          <p style={{ margin: '0 0 8px', fontSize: 12, color: '#888' }}>
            Don't have access?
          </p>
          <a
            href={REQUEST_FORM_URL}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: 13, color: '#2c3e50', fontWeight: 600,
              textDecoration: 'none', borderBottom: '1px solid #2c3e50',
            }}
          >
            Request access →
          </a>
        </div>
      </div>
    </div>
  );
}
