"use client";
import React, { useState } from 'react';
import Link from 'next/link';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      setIsSubmitted(true);
    }, 1500);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0A1216] relative overflow-hidden font-sans p-4">
      {/* Background glowing orbs */}
      <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-[#38F0B0]/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-[#2DAA7A]/10 blur-[120px] pointer-events-none" />

      {/* Main Card */}
      <div className="relative w-full max-w-[400px] z-10">
        <div className="bg-[#0F1A1A]/80 backdrop-blur-xl border border-[#38F0B0]/20 rounded-3xl p-8 shadow-[0_8px_32px_rgba(0,0,0,0.5)] shadow-[#38F0B0]/5 transition-all duration-500 hover:border-[#38F0B0]/40">
          
          <div className="text-center mb-8">
            <h1 className="text-3xl font-extrabold tracking-tight mb-2">
              <span className="text-[#3EE8B0]">AI </span>
              <span className="bg-gradient-to-br from-white to-[#B8F2D0] bg-clip-text text-transparent">BOWLER</span>
            </h1>
            <p className="text-[#A3C2B2] text-sm font-medium">Reset your password</p>
          </div>

          {isSubmitted ? (
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-[#38F0B0]/20 rounded-full flex items-center justify-center mb-4 border border-[#38F0B0]/50">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#38F0B0" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                  <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
              </div>
              <p className="text-center text-[#AACEC0] text-sm font-medium mb-8 leading-relaxed">
                If an account exists for <span className="text-white font-bold">{email}</span>, you will receive a password reset link shortly.
              </p>
              <Link href="/login" className="w-full bg-black/40 border border-[#2DAA7A]/50 hover:bg-[#142A24] hover:border-[#38F0B0] text-center text-white font-bold py-3.5 rounded-full transition-all duration-300">
                Return to Sign In
              </Link>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="flex flex-col gap-5">
              <p className="text-[#A3C2B2] text-sm text-center -mt-2 mb-2">
                Enter your email address and we'll send you a link to reset your password.
              </p>

              <div className="flex flex-col gap-1.5">
                <label className="text-[#AACEC0] text-xs font-semibold uppercase tracking-wider">Email Address</label>
                <input 
                  type="email" 
                  placeholder="bowler@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="w-full bg-black/40 border border-[#2DAA7A]/30 rounded-xl px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-[#38F0B0] focus:ring-1 focus:ring-[#38F0B0]/50 transition-all duration-300 shadow-inner"
                />
              </div>

              <button 
                type="submit" 
                disabled={isLoading}
                className={`mt-2 w-full bg-gradient-to-r from-[#142A24] to-[#1C4238] border border-[#38F0B0] hover:from-[#1C4238] hover:to-[#2DAA7A] text-white font-bold py-3.5 rounded-full transition-all duration-300 shadow-[0_0_20px_rgba(56,240,176,0.15)] hover:shadow-[0_0_30px_rgba(56,240,176,0.3)] transform hover:-translate-y-0.5 ${isLoading ? 'opacity-70 cursor-not-allowed transform-none' : ''}`}
              >
                {isLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5 text-[#38F0B0]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Sending...
                  </span>
                ) : 'Send Reset Link'}
              </button>
            </form>
          )}

          {!isSubmitted && (
            <div className="mt-8 pt-6 border-t border-white/5 text-center">
              <p className="text-[#7D9F8F] text-sm">
                Remember your password?{' '}
                <Link href="/login" className="text-[#38F0B0] font-semibold hover:text-white transition-colors">
                  Sign In
                </Link>
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
