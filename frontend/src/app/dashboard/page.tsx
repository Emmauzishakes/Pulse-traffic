"use client";
import React, { useState } from 'react';
import { useLiveNetwork } from '@/hooks/useTrafficData';
import { StatCard } from '@/components/StatCard';
import { Car, Activity, Map as MapIcon, Navigation, WifiOff, ArrowLeft } from 'lucide-react'; // Add WifiOff
import MapWrapper from '@/components/Map/MapWrapper';
import { DashboardCharts } from '@/components/Charts/DashboardCharts';
import Link from 'next/link';

export default function Dashboard() {
  // 1. Destructure the isError flag
  const { data: networkData, isLoading, isError } = useLiveNetwork();
  const [focusedNode, setFocusedNode] = useState<{lat: number, lng: number} | null>(null);

  // 2. Only show the full-screen loader if we have NO data at all (initial load)
  if (isLoading && !networkData) {
    return <div className="min-h-screen bg-slate-900 flex items-center justify-center text-pulse-cyan animate-pulse">Syncing with simulator...</div>;
  }

  const totalVehicles = networkData?.reduce((acc, curr) => acc + curr.vehicle_count, 0) || 0;
  const avgSpeed = networkData ? Math.round(networkData.reduce((acc, curr) => acc + curr.speed, 0) / networkData.length) : 0;
  const avgDensity = networkData ? Math.round((networkData.reduce((acc, curr) => acc + curr.density, 0) / networkData.length) * 100) : 0;
  const hotspots = networkData?.filter(d => d.status === 'Jammed').length || 0;

  return (
    <div className="min-h-screen bg-slate-900 text-white font-sans p-6">
      
      {/* NEW: 3. The Graceful Degradation Error Banner */}
      {isError && (
        <div className="mb-6 bg-pulse-red/10 border border-pulse-red/30 rounded-lg p-4 flex items-center gap-3 text-pulse-red">
          <WifiOff size={24} className="animate-pulse" />
          <div>
            <h3 className="font-semibold text-sm">Telemetry Feed Offline</h3>
            <p className="text-xs opacity-80 mt-0.5">
              Connection lost. Displaying last known data. Attempting to reconnect...
            </p>
          </div>
        </div>
      )}

      {/* Header Section */}
      <header className="mb-8">
        <Link 
          href="/" 
          className="group inline-flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 hover:border-pulse-cyan/50 rounded-full text-slate-400 hover:text-pulse-cyan text-xs font-medium transition-all mb-6 shadow-sm"
        >
          <ArrowLeft size={14} className="group-hover:-translate-x-0.5 transition-transform" />
          Return to Home
        </Link>
        <div className='flex justify-between items-end'>
          <div>
            <h1 className="text-2xl font-semibold mb-1">Nairobi Network • Live</h1>
            <p className="text-slate-400 text-sm">15 monitored junctions across 4 major corridors.</p>
          </div>
          <div className="flex items-center gap-4 text-sm">
            {/* Dynamically change the "Live" badge if offline */}
            <span className={`flex items-center gap-2 bg-slate-800 border px-3 py-1 rounded-full ${
              isError ? 'border-pulse-red/30 text-pulse-red' : 'border-pulse-green/30 text-pulse-green'
            }`}>
              <span className={`w-2 h-2 rounded-full ${isError ? 'bg-pulse-red' : 'bg-pulse-green animate-pulse'}`}></span> 
              {isError ? 'Offline' : 'Live'}
            </span>
          </div>
        </div>
      </header>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard title="Total Vehicles" value={totalVehicles.toLocaleString()} subtitle="across selection" Icon={Car} />
        <StatCard title="Avg Speed" value={`${avgSpeed} km/h`} subtitle={avgSpeed > 40 ? 'Free-flowing' : 'Congested'} Icon={Navigation} statusColor={avgSpeed > 40 ? 'green' : 'red'} />
        <StatCard title="Avg Density" value={`${avgDensity} v/km`} subtitle="lower is better" Icon={Activity} statusColor="cyan" />
        <StatCard title="Hotspots" value={hotspots} subtitle={`${networkData?.length || 0 - hotspots} clear`} Icon={MapIcon} statusColor={hotspots > 0 ? 'red' : 'green'} />
      </div>

      {/* Map & List Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-150">
        
        {/* Left: GIS Map */}
        <div className="lg:col-span-2 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden relative">
          <div className="absolute top-4 left-4 z-10 flex gap-2">
            <span className="bg-slate-900/80 px-3 py-1 rounded-full border border-pulse-green text-pulse-green text-xs font-bold flex items-center gap-2"><span className="w-2 h-2 bg-pulse-green rounded-full shadow-[0_0_8px_rgba(34,197,94,0.8)]"></span>Free</span>
            <span className="bg-slate-900/80 px-3 py-1 rounded-full border border-pulse-yellow text-pulse-yellow text-xs font-bold flex items-center gap-2"><span className="w-2 h-2 bg-pulse-yellow rounded-full shadow-[0_0_8px_rgba(234,179,8,0.8)]"></span>Slow</span>
            <span className="bg-slate-900/80 px-3 py-1 rounded-full border border-pulse-red text-pulse-red text-xs font-bold flex items-center gap-2"><span className="w-2 h-2 bg-pulse-red rounded-full shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span>Jammed</span>
          </div>
          <div className="w-full h-full relative z-0">
             <MapWrapper data={networkData || null} focusedNode={focusedNode} />
          </div>
        </div>

        {/* Right: Traffic Nodes List */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl flex flex-col overflow-hidden h-150">
          <div className="p-5 border-b border-slate-700 bg-slate-900/50">
            <h2 className="text-white font-semibold text-lg">Traffic Nodes</h2>
            <p className="text-slate-400 text-xs mt-1">{networkData?.length || 0} monitored intersections</p>
          </div>
          
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {networkData?.slice(0, 15).map((node) => (
              <div 
                key={node.id} 
                onClick={() => setFocusedNode({ lat: node.meta.lat, lng: node.meta.lng })}
                className="p-4 border-b border-slate-700/50 hover:bg-slate-700 transition-colors group cursor-pointer"
              >
                <div className="flex justify-between items-start mb-1">
                  <div className="flex items-center gap-3">
                    <span className={`w-2 h-2 rounded-full shadow-[0_0_8px_currentColor] ${
                      node.status === 'Free' ? 'bg-pulse-green text-pulse-green' : 
                      node.status === 'Slow' ? 'bg-pulse-yellow text-pulse-yellow' : 
                      'bg-pulse-red text-pulse-red'
                    }`}></span>
                    <h4 className="text-slate-200 font-medium text-sm group-hover:text-pulse-cyan transition-colors">
                      {node.meta.name}
                    </h4>
                  </div>
                  <span className={`text-[9px] font-bold tracking-wider uppercase px-2 py-0.5 rounded border ${
                    node.status === 'Free' ? 'text-slate-400 border-slate-600 bg-slate-900' : 
                    node.status === 'Slow' ? 'text-pulse-yellow border-pulse-yellow/30 bg-pulse-yellow/10' : 
                    'text-pulse-red border-pulse-red/30 bg-pulse-red/10'
                  }`}>
                    {node.status}
                  </span>
                </div>

                <p className="text-[10px] text-slate-400 uppercase tracking-widest ml-5 mb-3">
                  {node.meta.corridor}
                </p>

                <div className="flex justify-between text-xs text-slate-400 ml-5 pr-2">
                  <div className="flex flex-col">
                    <span className="text-slate-300 font-medium">{node.vehicle_count}</span>
                    <span className="text-[10px] text-slate-500">veh</span>
                  </div>
                  <div className="flex flex-col text-center">
                    <span className="text-slate-300 font-medium">{node.speed}</span>
                    <span className="text-[10px] text-slate-500">km/h</span>
                  </div>
                  <div className="flex flex-col text-right">
                    <span className="text-slate-300 font-medium">{Math.round(node.density * 100)}</span>
                    <span className="text-[10px] text-slate-500">v/km</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

      <DashboardCharts data={networkData || null} />
    </div>
  );
}