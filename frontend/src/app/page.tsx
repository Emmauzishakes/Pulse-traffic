import React from 'react';
import Link from 'next/link';
import { Activity, MapPin, Gauge, TrendingUp, Shield, ArrowRight } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-navy-950 text-white font-sans selection:bg-pulse-cyan/30">
      
      {/* Navbar Placeholder */}
      <nav className="flex justify-between items-center p-6 border-b border-gray-800 bg-navy-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-pulse-cyan flex items-center justify-center shadow-[0_0_10px_rgba(6,182,212,0.5)]">
            <Activity size={20} className="text-navy-950" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight">Pulse Traffic</h1>
            <p className="text-[10px] text-pulse-cyan uppercase tracking-wider">Nairobi • Live Corridor Intelligence</p>
          </div>
        </div>
        <div className="hidden md:flex gap-6 text-sm font-medium text-gray-400">
          <Link href="/" className="hover:text-white transition-colors">Home</Link>
          <Link href="/dashboard" className="text-white bg-navy-800 px-4 py-1.5 rounded-full border border-gray-700">Dashboard</Link>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <Link href="/dashboard" className="flex items-center gap-2 hover:text-pulse-cyan transition-colors">
            Go to App <ArrowRight size={16} />
          </Link>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 pt-20 pb-24">
        
        {/* Hero Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-32">
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-pulse-cyan/30 bg-pulse-cyan/10 text-pulse-cyan text-xs font-semibold tracking-wide uppercase">
              <Activity size={14} /> Live • 6 Nairobi Corridors
            </div>
            
            <h2 className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1]">
              Feel the city's <span className="text-transparent bg-clip-text bg-gradient-to-r from-pulse-cyan to-blue-400 drop-shadow-[0_0_15px_rgba(6,182,212,0.4)]">pulse</span><br />
              before you move.
            </h2>
            
            <p className="text-gray-400 text-lg md:text-xl leading-relaxed max-w-xl">
              Pulse Traffic turns Nairobi's busiest roads — Thika Superhighway, Ngong Road, Waiyaki Way, Mombasa Road, Jogoo Road and Lang'ata Road — into a live, color-coded map with 2-hour predictive analytics. Leave at the right time, every time.
            </p>
            
            <div className="flex flex-wrap items-center gap-4 pt-4">
              <Link href="/dashboard" className="bg-pulse-cyan text-navy-950 px-6 py-3 rounded-lg font-semibold hover:bg-cyan-400 transition-all shadow-[0_0_20px_rgba(6,182,212,0.4)] flex items-center gap-2">
                Launch Dashboard <ArrowRight size={18} />
              </Link>
              <button className="px-6 py-3 rounded-lg font-semibold border border-gray-700 hover:bg-navy-800 transition-colors">
                View Documentation
              </button>
            </div>

            <div className="flex gap-4 pt-8 border-t border-gray-800/50">
              <span className="flex items-center gap-2 text-sm text-gray-300"><span className="w-3 h-3 rounded-full bg-pulse-green shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span> Free</span>
              <span className="flex items-center gap-2 text-sm text-gray-300"><span className="w-3 h-3 rounded-full bg-pulse-yellow shadow-[0_0_8px_rgba(234,179,8,0.6)]"></span> Slow</span>
              <span className="flex items-center gap-2 text-sm text-gray-300"><span className="w-3 h-3 rounded-full bg-pulse-red shadow-[0_0_8px_rgba(239,68,68,0.6)]"></span> Jammed</span>
            </div>
          </div>

          {/* Hero Right: "City Pulse" Mini Dashboard Graphic */}
          <div className="relative">
            {/* Background ambient glow */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-pulse-cyan/10 blur-[100px] rounded-full pointer-events-none"></div>
            
            <div className="bg-navy-900 border border-gray-700/50 rounded-2xl p-6 shadow-2xl relative z-10">
              <div className="flex justify-between items-start mb-8">
                <div>
                  <p className="text-gray-500 text-xs tracking-widest uppercase mb-1">Live • Nairobi Network</p>
                  <h3 className="text-2xl font-semibold">City pulse</h3>
                </div>
                <div className="bg-pulse-green/10 border border-pulse-green/30 text-pulse-green px-3 py-1 rounded-full text-xs animate-pulse">
                  Improving
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-8">
                <div className="bg-navy-950 p-4 rounded-xl border border-gray-800">
                  <p className="text-gray-500 text-xs mb-1">SPEED</p>
                  <p className="text-2xl font-bold text-pulse-green">41 <span className="text-sm font-normal text-gray-500">km/h</span></p>
                </div>
                <div className="bg-navy-950 p-4 rounded-xl border border-gray-800">
                  <p className="text-gray-500 text-xs mb-1">DENSITY</p>
                  <p className="text-2xl font-bold text-pulse-yellow">44 <span className="text-sm font-normal text-gray-500">v/km</span></p>
                </div>
                <div className="bg-navy-950 p-4 rounded-xl border border-gray-800">
                  <p className="text-gray-500 text-xs mb-1">HOTSPOTS</p>
                  <p className="text-2xl font-bold text-pulse-red">5</p>
                </div>
              </div>

              <div className="space-y-4">
                {[
                  { name: 'Globe Roundabout • Thika Hwy', color: 'bg-pulse-red', width: 'w-full' },
                  { name: 'Westlands • Waiyaki Way', color: 'bg-pulse-red', width: 'w-[85%]' },
                  { name: 'Nyayo Stadium • Mombasa Rd', color: 'bg-pulse-yellow', width: 'w-[60%]' },
                  { name: 'Junction Mall • Ngong Rd', color: 'bg-pulse-yellow', width: 'w-[50%]' },
                  { name: 'Bomas • Lang\'ata Rd', color: 'bg-pulse-green', width: 'w-[20%]' },
                ].map((node, i) => (
                  <div key={i} className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3 w-1/2">
                      <span className={`w-2 h-2 rounded-full ${node.color}`}></span>
                      <span className="text-sm text-gray-300 truncate">{node.name}</span>
                    </div>
                    <div className="w-1/2 h-1.5 bg-navy-950 rounded-full overflow-hidden">
                      <div className={`h-full ${node.color} ${node.width}`}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="mb-32">
          <div className="mb-12">
            <h2 className="text-3xl font-bold mb-2">Built for the Nairobi commute.</h2>
            <p className="text-gray-400">From CBD bottlenecks to the suburbs — every interchange, every minute.</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-navy-900 border border-gray-800 p-6 rounded-xl hover:border-gray-600 transition-colors">
              <MapPin className="text-pulse-cyan mb-4" size={28} />
              <h3 className="text-lg font-semibold mb-2">Live GIS map</h3>
              <p className="text-gray-400 text-sm leading-relaxed">Leaflet-powered dark map with color-coded nodes across six corridors.</p>
            </div>
            <div className="bg-navy-900 border border-gray-800 p-6 rounded-xl hover:border-gray-600 transition-colors">
              <Gauge className="text-pulse-green mb-4" size={28} />
              <h3 className="text-lg font-semibold mb-2">Real-time metrics</h3>
              <p className="text-gray-400 text-sm leading-relaxed">Vehicle counts, speeds and density updated every few seconds.</p>
            </div>
            <div className="bg-navy-900 border border-gray-800 p-6 rounded-xl hover:border-gray-600 transition-colors">
              <TrendingUp className="text-pulse-yellow mb-4" size={28} />
              <h3 className="text-lg font-semibold mb-2">2-hour forecasts</h3>
              <p className="text-gray-400 text-sm leading-relaxed">Predictive congestion modelling so you know exactly when to leave.</p>
            </div>
            <div className="bg-navy-900 border border-gray-800 p-6 rounded-xl hover:border-gray-600 transition-colors">
              <Shield className="text-pulse-red mb-4" size={28} />
              <h3 className="text-lg font-semibold mb-2">Operator-grade</h3>
              <p className="text-gray-400 text-sm leading-relaxed">Hotspot ranking and historical trends for planners and dispatch.</p>
            </div>
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-navy-800 to-navy-900 border border-gray-700 p-12 text-center">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-pulse-cyan/20 via-transparent to-transparent pointer-events-none"></div>
          
          <Activity className="mx-auto text-pulse-cyan mb-6" size={48} />
          <h2 className="text-3xl font-bold mb-4">Ready to skip the jam?</h2>
          <p className="text-gray-400 mb-8 max-w-md mx-auto">Access the live dashboard immediately. Connects directly to the backend simulator pipeline.</p>
          
          <Link href="/dashboard" className="inline-flex items-center gap-2 bg-pulse-cyan text-navy-950 px-8 py-3 rounded-lg font-semibold hover:bg-cyan-400 transition-all shadow-[0_0_20px_rgba(6,182,212,0.4)]">
            Go to Dashboard <ArrowRight size={18} />
          </Link>
        </div>

      </main>

      {/* Footer */}
      <footer className="text-center py-8 text-gray-500 text-sm border-t border-gray-800">
        <p>© 2026 Pulse Traffic • Simulated data for STQA demo purposes.</p>
      </footer>
    </div>
  );
}