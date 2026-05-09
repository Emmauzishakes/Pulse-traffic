"use client";
import React, { useState } from 'react';
import Link from 'next/link';
import { Activity, MapPin, Gauge, TrendingUp, Shield, ArrowRight, WifiOff, Bell, Mail, CheckCircle } from 'lucide-react';
import { useLiveNetwork } from '@/hooks/useTrafficData';

export default function HomePage() {
  // Fetch live data from the backend
  const { data: networkData, isLoading, isError } = useLiveNetwork();

  const [ subscribed, setSubscribed ] = useState(false);
  const [ email, setEmail ] = useState('');

  const handleSubscribe = ( e: React.FormEvent ) => {
    e.preventDefault();
    if (email) {
      setSubscribed(true);
      setTimeout(() => setSubscribed(false), 3000);
      setEmail('');
    }
  };

  // Calculate dynamic metrics
  const avgSpeed = networkData && networkData.length > 0 
    ? Math.round(networkData.reduce((acc, curr) => acc + curr.speed, 0) / networkData.length) 
    : 0;
  
  const avgDensity = networkData && networkData.length > 0
    ? Math.round((networkData.reduce((acc, curr) => acc + curr.density, 0) / networkData.length) * 100) 
    : 0;
  
  const hotspots = networkData?.filter(d => d.status === 'Jammed').length || 0;

  // Grab up to 5 nodes to display in the live preview
  const previewNodes = networkData?.slice(0, 5) || [];

  return (
    <div className="min-h-screen w-[100vw] bg-slate-900 text-white font-sans selection:bg-pulse-cyan/30 overflow-x-hidden">
      
      {/* Navbar */}
      <nav className="flex justify-between items-center p-6 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-pulse-cyan flex items-center justify-center shadow-[0_0_10px_rgba(6,182,212,0.5)]">
            <Activity size={20} className="text-slate-900" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight">Pulse Traffic</h1>
            <p className="text-[10px] text-pulse-cyan uppercase tracking-wider">Nairobi • Live Corridor Intelligence</p>
          </div>
        </div>
        <div className="hidden md:flex gap-6 text-sm font-medium text-slate-400">
          <Link href="/" className="text-white bg-slate-800 px-4 py-1.5 rounded-full border border-slate-700 transition-colors">Home</Link>
          <Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <Link href="/dashboard" className="flex items-center gap-2 hover:text-pulse-cyan transition-colors">
            Go to App <ArrowRight size={16} />
          </Link>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 pt-5 pb-24">
        
        {/* Hero Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-32">
          <div className="space-y-8">
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-semibold tracking-wide uppercase ${
              isError ? 'border-pulse-red/30 bg-pulse-red/10 text-pulse-red' : 'border-pulse-cyan/30 bg-pulse-cyan/10 text-pulse-cyan'
            }`}>
              {isError ? <WifiOff size={14} /> : <Activity size={14} />} 
              {isError ? 'Simulator Offline' : 'Live • 4 Nairobi Corridors'}
            </div>
            
            <h2 className="text-5xl md:text-5xl font-bold tracking-tight leading-[1.1]">
              Feel the city&apos;s <span className="text-transparent bg-clip-text bg-linear-to-r from-pulse-cyan to-blue-400 drop-shadow-[0_0_15px_rgba(6,182,212,0.4)]">pulse</span><br />
              before you move.
            </h2>
            
            <p className="text-slate-400 text-lg md:text-xl leading-relaxed max-w-xl">
              Pulse Traffic turns Nairobi&apos;s busiest arteries — Thika Superhighway, The Nairobi Expressway, Ngong Road, and Lang&apos;ata Road — into a live, color-coded map with real-time telemetry. Leave at the right time, every time.
            </p>
            
            <div className="flex flex-wrap items-center gap-4 pt-4">
              <Link href="/dashboard" className="bg-pulse-cyan text-slate-900 px-6 py-3 rounded-lg font-bold hover:bg-cyan-400 transition-all shadow-[0_0_20px_rgba(6,182,212,0.4)] flex items-center gap-2">
                Launch Dashboard <ArrowRight size={18} />
              </Link>
              {/* <button className="px-6 py-3 rounded-lg font-semibold border border-slate-700 hover:bg-slate-800 transition-colors">
                View Architecture
              </button> */}
              <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" className="px-6 py-3 rounded-lg font-semibold border border-slate-700 hover:bg-slate-800 transition-colors">
                View Architecture
              </a>
            </div>

            <div className="flex gap-4 pt-8 border-t border-slate-800/50">
              <span className="flex items-center gap-2 text-sm text-slate-300"><span className="w-3 h-3 rounded-full bg-pulse-green shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span> Free</span>
              <span className="flex items-center gap-2 text-sm text-slate-300"><span className="w-3 h-3 rounded-full bg-pulse-yellow shadow-[0_0_8px_rgba(234,179,8,0.6)]"></span> Slow</span>
              <span className="flex items-center gap-2 text-sm text-slate-300"><span className="w-3 h-3 rounded-full bg-pulse-red shadow-[0_0_8px_rgba(239,68,68,0.6)]"></span> Jammed</span>
            </div>
          </div>

          {/* Hero Right: "City Pulse" Mini Dashboard Graphic */}
          <div className="relative">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-pulse-cyan/10 blur-[100px] rounded-full pointer-events-none"></div>
            
            <div className="bg-slate-800 border border-slate-700/50 rounded-2xl p-6 shadow-2xl relative z-10">
              <div className="flex justify-between items-start mb-8">
                <div>
                  <p className="text-slate-400 text-xs tracking-widest uppercase mb-1">Live • Nairobi Network</p>
                  <h3 className="text-2xl font-semibold">City pulse</h3>
                </div>
                {!isError && (
                  <div className="bg-pulse-green/10 border border-pulse-green/30 text-pulse-green px-3 py-1 rounded-full text-xs animate-pulse">
                    Live Feed Active
                  </div>
                )}
              </div>

              <div className="grid grid-cols-3 gap-4 mb-8">
                <div className="bg-slate-900 p-4 rounded-xl border border-slate-700">
                  <p className="text-slate-400 text-xs mb-1">AVG SPEED</p>
                  <p className={`text-2xl font-bold ${avgSpeed > 40 ? 'text-pulse-green' : 'text-pulse-yellow'}`}>
                    {isLoading ? '--' : avgSpeed} <span className="text-sm font-normal text-slate-500">km/h</span>
                  </p>
                </div>
                <div className="bg-slate-900 p-4 rounded-xl border border-slate-700">
                  <p className="text-slate-400 text-xs mb-1">AVG DENSITY</p>
                  <p className="text-2xl font-bold text-pulse-cyan">
                    {isLoading ? '--' : avgDensity} <span className="text-sm font-normal text-slate-500">v/km</span>
                  </p>
                </div>
                <div className="bg-slate-900 p-4 rounded-xl border border-slate-700">
                  <p className="text-slate-400 text-xs mb-1">HOTSPOTS</p>
                  <p className={`text-2xl font-bold ${hotspots > 0 ? 'text-pulse-red' : 'text-pulse-green'}`}>
                    {isLoading ? '--' : hotspots}
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                {isLoading ? (
                  <div className="text-center text-slate-500 text-sm py-4 animate-pulse">Connecting to IoT Simulator...</div>
                ) : previewNodes.length > 0 ? (
                  previewNodes.map((node, i) => {
                    const color = node.status === 'Free' ? 'bg-pulse-green' : node.status === 'Slow' ? 'bg-pulse-yellow' : 'bg-pulse-red';
                    // Convert density (0.0 to 1.0) to a percentage for the bar width
                    const widthPercent = Math.min(Math.max(node.density * 100, 10), 100); 
                    
                    return (
                      <div key={node.id || i} className="flex items-center justify-between gap-4">
                        <div className="flex items-center gap-3 w-1/2">
                          <span className={`w-2 h-2 rounded-full ${color} shadow-[0_0_8px_currentColor]`}></span>
                          <span className="text-sm text-slate-300 truncate">{node.meta?.name || node.node_id}</span>
                        </div>
                        <div className="w-1/2 h-1.5 bg-slate-900 rounded-full overflow-hidden">
                          <div className={`h-full ${color} transition-all duration-1000`} style={{ width: `${widthPercent}%` }}></div>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-center text-slate-500 text-sm py-4">No data flowing. Start the simulator.</div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="mb-32">
          <div className="mb-12">
            <h2 className="text-3xl font-bold mb-2">Built for the Nairobi commute.</h2>
            <p className="text-slate-400">From CBD bottlenecks to the suburbs — every interchange monitored.</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-slate-800 border border-slate-700 p-6 rounded-xl hover:border-slate-500 transition-colors">
              <MapPin className="text-pulse-cyan mb-4" size={28} />
              <h3 className="text-lg font-semibold mb-2">Live GIS Map</h3>
              <p className="text-slate-400 text-sm leading-relaxed">Leaflet-powered dark map mapping 15 critical nodes across four major highways.</p>
            </div>
            <div className="bg-slate-800 border border-slate-700 p-6 rounded-xl hover:border-slate-500 transition-colors">
              <Gauge className="text-pulse-green mb-4" size={28} />
              <h3 className="text-lg font-semibold mb-2">Real-Time Metrics</h3>
              <p className="text-slate-400 text-sm leading-relaxed">IoT-simulated vehicle counts, speeds, and congestion density updated seamlessly.</p>
            </div>
            <div className="bg-slate-800 border border-slate-700 p-6 rounded-xl hover:border-slate-500 transition-colors">
              <TrendingUp className="text-pulse-yellow mb-4" size={28} />
              <h3 className="text-lg font-semibold mb-2">Data Analytics</h3>
              <p className="text-slate-400 text-sm leading-relaxed">Historical trends and aggregated telemetry visualization using Recharts.</p>
            </div>
            <div className="bg-slate-800 border border-slate-700 p-6 rounded-xl hover:border-slate-500 transition-colors">
              <Shield className="text-pulse-red mb-4" size={28} />
              <h3 className="text-lg font-semibold mb-2">STQA Validated</h3>
              <p className="text-slate-400 text-sm leading-relaxed">Architected with robust fault-injection testing, mocking, and API validation.</p>
            </div>
          </div>
        </div>

        {/* Mock Email Alert Form */}
        <div className="mb-32 bg-slate-800 border border-slate-700 rounded-2xl p-8 md:p-12 flex flex-col md:flex-row items-center justify-between gap-8 relative overflow-hidden shadow-xl">
          <div className="absolute -right-20 -top-20 w-64 h-64 bg-pulse-cyan/5 blur-[80px] rounded-full pointer-events-none"></div>
          
          <div className="flex items-start gap-5 relative z-10">
            <div className="bg-pulse-cyan/10 p-3.5 rounded-xl text-pulse-cyan border border-pulse-cyan/20">
              <Bell size={28} />
            </div>
            <div>
              <h3 className="text-2xl font-bold mb-2">Get instant alerts when your route is jammed.</h3>
              <p className="text-slate-400 text-sm max-w-md">Stay ahead of the gridlock. Receive a live email notification the moment congestion exceeds threshold limits on your daily commute.</p>
            </div>
          </div>

          <form onSubmit={handleSubscribe} className="w-full md:w-auto flex flex-col sm:flex-row gap-3 relative z-10">
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
              <input
                type="email"
                placeholder="Enter your email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full sm:w-72 bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-3 text-sm focus:outline-none focus:border-pulse-cyan focus:ring-1 focus:ring-pulse-cyan transition-all"
                required
              />
            </div>
            <button
              type="submit"
              className={`px-6 py-3 rounded-lg font-bold transition-all flex items-center justify-center gap-2 ${
                subscribed
                  ? 'bg-pulse-green text-slate-900 shadow-[0_0_15px_rgba(34,197,94,0.4)]'
                  : 'bg-slate-700 text-white hover:bg-pulse-cyan hover:text-slate-900'
              }`}
            >
              {subscribed ? <><CheckCircle size={18} /> Subscribed</> : 'Subscribe'}
            </button>
          </form>
        </div>

        {/* Bottom CTA */}
        <div className="relative overflow-hidden rounded-2xl bg-linear-to-br from-slate-800 to-slate-900 border border-slate-700 p-12 text-center">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-pulse-cyan/20 via-transparent to-transparent pointer-events-none"></div>
          
          <Activity className="mx-auto text-pulse-cyan mb-6" size={48} />
          <h2 className="text-3xl font-bold mb-4">Ready to skip the jam?</h2>
          <p className="text-slate-400 mb-8 max-w-md mx-auto">Access the live dashboard immediately. Connects directly to the backend IoT simulator pipeline.</p>
          
          <Link href="/dashboard" className="inline-flex items-center gap-2 bg-pulse-cyan text-slate-900 px-8 py-3 rounded-lg font-bold hover:bg-cyan-400 transition-all shadow-[0_0_20px_rgba(6,182,212,0.4)]">
            Go to Dashboard <ArrowRight size={18} />
          </Link>
        </div>

      </main>

      {/* Footer */}
      <footer className="text-center py-8 text-slate-500 text-sm border-t border-slate-800">
        <p>© {new Date().getFullYear()} Pulse Traffic • Software Testing & Quality Assurance Architecture.</p>
      </footer>
    </div>
  );
}