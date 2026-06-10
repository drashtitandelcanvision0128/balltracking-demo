"use client";
import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';

interface Player {
  id: string;
  name: string;
  role: string;
  style: string;
}

const CustomSelect = ({ label, value, options, onChange, dropdownPosition = 'bottom' }: { label?: string, value: string, options: string[], onChange: (val: string) => void, dropdownPosition?: 'bottom' | 'top' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  return (
    <div className="flex flex-col gap-1.5 relative w-full" ref={dropdownRef}>
      {label && <label className="text-[#AACEC0] text-xs font-semibold uppercase tracking-wider">{label}</label>}
      <div 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full bg-black/40 border border-[#2DAA7A]/30 rounded-xl px-4 py-3 text-white cursor-pointer flex justify-between items-center transition-all duration-300 shadow-inner hover:border-[#38F0B0]/50"
      >
        <span>{value}</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`transition-transform duration-300 ${isOpen ? 'rotate-180 text-[#38F0B0]' : 'text-[#AACEC0]'}`}>
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </div>
      {isOpen && (
        <div className={`absolute left-0 w-full bg-[#0F1A1A] border border-[#2DAA7A]/50 rounded-xl overflow-y-auto max-h-48 z-50 shadow-[0_8px_32px_rgba(0,0,0,0.8)] backdrop-blur-xl [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[#2DAA7A]/50 hover:[&::-webkit-scrollbar-thumb]:bg-[#2DAA7A] [&::-webkit-scrollbar-thumb]:rounded-full ${dropdownPosition === 'top' ? 'bottom-[100%] mb-2' : 'top-[100%] mt-2'}`}>
          {options.map((opt) => (
            <div 
              key={opt}
              onClick={() => { onChange(opt); setIsOpen(false); }}
              className={`px-4 py-3 hover:bg-[#38F0B0]/20 cursor-pointer transition-colors text-sm font-medium ${value === opt ? 'text-[#38F0B0] bg-[#38F0B0]/10' : 'text-[#E8F0EC]'}`}
            >
              {opt}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default function PlayersDashboard() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [hasMounted, setHasMounted] = useState(false);
  
  // Form states
  const [showForm, setShowForm] = useState(false);
  const [editingPlayerId, setEditingPlayerId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [name, setName] = useState('');
  const [role, setRole] = useState('Batsman');
  const [style, setStyle] = useState('Right-handed');

  useEffect(() => {
    setHasMounted(true);
    if (typeof window !== 'undefined') {
      const loggedIn = localStorage.getItem('userLoggedIn') === 'true';
      setIsLoggedIn(loggedIn);
      
      if (loggedIn) {
        const savedPlayers = localStorage.getItem('my_players');
        if (savedPlayers) {
          setPlayers(JSON.parse(savedPlayers));
        }
      } else {
        window.location.href = '/login';
      }
    }
  }, []);

  const savePlayers = (newPlayers: Player[]) => {
    setPlayers(newPlayers);
    localStorage.setItem('my_players', JSON.stringify(newPlayers));
  };

  const handleSubmitPlayer = (e: React.FormEvent) => {
    e.preventDefault();
    if (editingPlayerId) {
      const updatedPlayers = players.map(p => 
        p.id === editingPlayerId 
          ? { ...p, name, role, style } 
          : p
      );
      savePlayers(updatedPlayers);
      setEditingPlayerId(null);
    } else {
      const newPlayer: Player = {
        id: Date.now().toString(),
        name,
        role,
        style
      };
      savePlayers([...players, newPlayer]);
    }
    setName('');
    setShowForm(false);
  };

  const handleEditClick = (player: Player) => {
    setEditingPlayerId(player.id);
    setName(player.name);
    setRole(player.role);
    setStyle(player.style);
    setShowForm(true);
  };

  const handleDeletePlayer = (id: string) => {
    const newPlayers = players.filter(p => p.id !== id);
    savePlayers(newPlayers);
  };

  const handleSelectPlayer = (player: Player) => {
    localStorage.setItem('selectedPlayer', JSON.stringify(player));
    window.location.href = '/';
  };

  useEffect(() => {
    const maxPage = Math.max(1, Math.ceil(players.length / itemsPerPage));
    if (currentPage > maxPage) {
      setCurrentPage(maxPage);
    }
  }, [players.length, itemsPerPage, currentPage]);

  const totalPages = Math.ceil(players.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentPlayers = players.slice(startIndex, startIndex + itemsPerPage);

  if (!hasMounted || !isLoggedIn) return null;

  return (
    <div className="min-h-screen bg-[#0A1216] relative overflow-hidden font-sans text-white p-6">
      {/* Background glowing orbs */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-[#38F0B0]/10 blur-[150px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-[#2DAA7A]/10 blur-[150px] pointer-events-none" />

      <div className="max-w-5xl mx-auto relative z-10">
        <header className="flex justify-between items-center mb-10 border-b border-[#38F0B0]/20 pb-6 mt-4">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight">
              <span className="text-[#3EE8B0]">MY </span>
              <span className="bg-gradient-to-br from-white to-[#B8F2D0] bg-clip-text text-transparent">PLAYERS</span>
            </h1>
            <p className="text-[#A3C2B2] text-sm mt-1">Manage your team and track performance</p>
          </div>
          <div className="flex gap-4">
            <Link href="/" className="px-5 py-2.5 rounded-full border border-[#2DAA7A] text-[#C6F0DE] hover:bg-[#142A24] transition-colors font-semibold text-sm flex items-center gap-2">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
              </svg>
              Dashboard
            </Link>
            <button 
              onClick={() => {
                setEditingPlayerId(null);
                setName('');
                setRole('Batsman');
                setStyle('Right-handed');
                setShowForm(true);
              }}
              className="px-5 py-2.5 rounded-full bg-gradient-to-r from-[#2DAA7A] to-[#38F0B0] text-[#0A1216] font-bold text-sm hover:opacity-90 transition-opacity shadow-[0_0_15px_rgba(56,240,176,0.3)] flex items-center gap-2"
            >
              + Add Player
            </button>
          </div>
        </header>

        {showForm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
            <div className="bg-[#0F1A1A] border border-[#38F0B0]/30 rounded-2xl p-6 w-full max-w-sm shadow-[0_0_40px_rgba(56,240,176,0.15)] relative">
              <button 
                onClick={() => {
                  setShowForm(false);
                  setEditingPlayerId(null);
                  setName('');
                  setRole('Batsman');
                  setStyle('Right-handed');
                }}
                className="absolute top-4 right-4 text-[#AACEC0] hover:text-[#E04545] transition-colors p-1"
                title="Close"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
              <h2 className="text-xl font-bold text-[#38F0B0] mb-6">{editingPlayerId ? 'Edit Player' : 'Add New Player'}</h2>
              <form onSubmit={handleSubmitPlayer} className="flex flex-col gap-5">
                <div className="flex flex-col gap-1.5">
                  <label className="text-[#AACEC0] text-xs font-semibold uppercase tracking-wider">Player Name</label>
                  <input 
                    type="text" 
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    placeholder="e.g. Sachin Tendulkar"
                    className="w-full bg-black/40 border border-[#2DAA7A]/30 rounded-xl px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-[#38F0B0] focus:ring-1 focus:ring-[#38F0B0]/50 transition-all duration-300 shadow-inner"
                  />
                </div>
                <CustomSelect 
                  label="Role" 
                  value={role} 
                  options={["Batsman", "Bowler", "All-rounder", "Wicket-keeper"]} 
                  onChange={(val) => setRole(val)} 
                />
                <CustomSelect 
                  label="Style" 
                  value={style} 
                  options={["Right-handed", "Left-handed", "Right-arm Fast", "Right-arm Spin", "Left-arm Fast", "Left-arm Spin"]} 
                  onChange={(val) => setStyle(val)} 
                />
                <button 
                  type="submit"
                  className="mt-4 bg-gradient-to-r from-[#142A24] to-[#1C4238] border border-[#38F0B0] hover:from-[#1C4238] hover:to-[#2DAA7A] text-white font-bold py-3.5 rounded-xl transition-all duration-300 shadow-[0_0_20px_rgba(56,240,176,0.15)] hover:shadow-[0_0_30px_rgba(56,240,176,0.3)]"
                >
                  {editingPlayerId ? 'Update Player' : 'Save Player'}
                </button>
              </form>
            </div>
          </div>
        )}

        {players.length === 0 ? (
          <div className="bg-[#0F1A1A]/50 border border-dashed border-[#2DAA7A]/50 rounded-3xl p-12 text-center mt-10">
            <div className="w-20 h-20 mx-auto bg-[#142A24] rounded-full flex items-center justify-center mb-6">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#38F0B0" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">No players added yet</h3>
            <p className="text-[#A3C2B2] mb-8 max-w-md mx-auto">Start building your team by adding your first player. Track their batting or bowling trajectories easily.</p>
            <button 
              onClick={() => setShowForm(true)}
              className="px-8 py-3.5 rounded-full bg-[#142A24] border border-[#38F0B0] text-[#38F0B0] font-bold text-sm hover:bg-[#1C4238] transition-colors"
            >
              Add First Player
            </button>
          </div>
        ) : (
          <>
            <div className="flex justify-end mb-6">
              <div className="bg-[#0F1A1A] p-1 rounded-xl border border-[#38F0B0]/20 flex gap-1">
                <button 
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded-lg transition-colors ${viewMode === 'grid' ? 'bg-[#38F0B0]/20 text-[#38F0B0]' : 'text-[#AACEC0] hover:text-white'}`}
                  title="Grid View"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="7" height="7"></rect>
                    <rect x="14" y="3" width="7" height="7"></rect>
                    <rect x="14" y="14" width="7" height="7"></rect>
                    <rect x="3" y="14" width="7" height="7"></rect>
                  </svg>
                </button>
                <button 
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded-lg transition-colors ${viewMode === 'list' ? 'bg-[#38F0B0]/20 text-[#38F0B0]' : 'text-[#AACEC0] hover:text-white'}`}
                  title="List View"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="8" y1="6" x2="21" y2="6"></line>
                    <line x1="8" y1="12" x2="21" y2="12"></line>
                    <line x1="8" y1="18" x2="21" y2="18"></line>
                    <line x1="3" y1="6" x2="3.01" y2="6"></line>
                    <line x1="3" y1="12" x2="3.01" y2="12"></line>
                    <line x1="3" y1="18" x2="3.01" y2="18"></line>
                  </svg>
                </button>
              </div>
            </div>

            <div className={viewMode === 'grid' ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" : "flex flex-col gap-4"}>
              {currentPlayers.map(player => (
                <div 
                  key={player.id} 
                  onClick={() => handleSelectPlayer(player)}
                  className={`bg-[#0F1A1A] border border-[#38F0B0]/20 rounded-2xl p-6 hover:border-[#38F0B0]/50 transition-all duration-300 group relative hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(56,240,176,0.1)] cursor-pointer ${viewMode === 'list' ? 'flex items-center justify-between' : ''}`}
                >
                  <div className={`absolute transition-opacity opacity-0 group-hover:opacity-100 flex gap-2 z-10 ${viewMode === 'list' ? 'right-6 top-1/2 -translate-y-1/2' : 'top-4 right-4'}`}>
                    <button 
                      onClick={(e) => { e.stopPropagation(); handleEditClick(player); }}
                      className="text-[#38F0B0] bg-[#142A24] p-2 rounded-full hover:bg-[#1C4238]"
                      title="Edit Player"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                      </svg>
                    </button>
                    <button 
                      onClick={(e) => { e.stopPropagation(); handleDeletePlayer(player.id); }}
                      className="text-[#E04545] bg-[#2A1414] p-2 rounded-full hover:bg-[#3A1818]"
                      title="Delete Player"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                      </svg>
                    </button>
                  </div>
                  <div className={`flex items-center gap-4 ${viewMode === 'grid' ? 'mb-5' : ''}`}>
                    <div className="w-14 h-14 rounded-full bg-gradient-to-br from-[#2DAA7A] to-[#142A24] flex items-center justify-center text-xl font-bold text-white shadow-[0_0_15px_rgba(56,240,176,0.2)] border border-[#38F0B0]/30 shrink-0">
                      {player.name.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-white">{player.name}</h3>
                      <p className="text-[#38F0B0] text-sm font-semibold">{player.role}</p>
                    </div>
                  </div>
                  <div className={`bg-[#0A1216] rounded-xl p-3 border border-white/5 ${viewMode === 'list' ? 'mr-28 min-w-[200px]' : ''}`}>
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-[#A3C2B2] uppercase tracking-wider font-semibold mr-4">Style</span>
                      <span className="text-white font-medium">{player.style}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {players.length > 0 && (
              <div className="flex justify-between items-center mt-8 p-4 bg-[#0F1A1A]/50 border border-[#38F0B0]/20 rounded-xl flex-wrap gap-4">
                <div className="flex items-center gap-3">
                  <span className="text-[#AACEC0] text-sm font-medium">Show:</span>
                  <div className="w-20">
                    <CustomSelect 
                      value={itemsPerPage.toString()} 
                      options={["3", "6", "12", "24"]} 
                      onChange={(val) => {
                        setItemsPerPage(parseInt(val));
                        setCurrentPage(1);
                      }} 
                      dropdownPosition="top"
                    />
                  </div>
                  <span className="text-[#AACEC0] text-sm font-medium">entries</span>
                </div>

                {totalPages > 1 && (
                  <div className="flex items-center gap-4">
                    <button 
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="px-5 py-2 rounded-lg bg-[#142A24] border border-[#38F0B0]/30 text-[#38F0B0] font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#1C4238] transition-colors flex items-center gap-2"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="15 18 9 12 15 6"></polyline>
                      </svg>
                      Prev
                    </button>
                    <div className="text-[#A3C2B2] text-sm font-medium">
                      Page <span className="text-white mx-1">{currentPage}</span> of <span className="text-white mx-1">{totalPages}</span>
                    </div>
                    <button 
                      onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="px-5 py-2 rounded-lg bg-[#142A24] border border-[#38F0B0]/30 text-[#38F0B0] font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#1C4238] transition-colors flex items-center gap-2"
                    >
                      Next
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="9 18 15 12 9 6"></polyline>
                      </svg>
                    </button>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
