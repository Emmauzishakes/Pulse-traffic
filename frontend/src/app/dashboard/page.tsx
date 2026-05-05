"use client";
import React from 'react';
import { useLiveNetwork } from '@/hooks/useTrafficData';
import { StatCard } from '@/components/StatCard';
import { Car, Activity, Map as MapIcon, Navigation } from 'lucide-react';
import MapWrapper from '@/components/Map/MapWrapper';
import { DashboardCharts } from '@/components/Charts/DashboardCharts';

export default function Dashboard() {
  const { data: networkData, isLoading } = useLiveNetwork();

  if (isLoading) return <div className="min-h-screen bg-navy-950 flex items-center justify-center text-pulse-cyan">Syncing with simulator...</div>;

  // Calculate Aggregates
  const totalVehicles = networkData?.reduce((acc, curr) => acc + curr.vehicle_count, 0) || 0;
  const avgSpeed = networkData ? Math.round(networkData.reduce((acc, curr) => acc + curr.speed, 0) / networkData.length) : 0;
  const avgDensity = networkData ? Math.round((networkData.reduce((acc, curr) => acc + curr.density, 0) / networkData.length) * 100) : 0;
  const hotspots = networkData?.filter(d => d.status === 'Jammed').length || 0;

  return (
    <div className="min-h-screen bg-navy-950 text-white font-sans p-6">
      {/* Header Section */}
      <header className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-2xl font-semibold mb-1">Nairobi Network • Live</h1>
          <p className="text-gray-400 text-sm">39 monitored junctions across 6 corridors.</p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span className="flex items-center gap-2 bg-navy-900 border border-pulse-green/30 px-3 py-1 rounded-full text-pulse-green">
            <span className="w-2 h-2 rounded-full bg-pulse-green animate-pulse"></span> Live
          </span>
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
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[600px]">
        
        {/* Left: GIS Map */}
        <div className="lg:col-span-2 bg-navy-900 border border-gray-800 rounded-xl overflow-hidden relative">
          <div className="absolute top-4 left-4 z-10 flex gap-2">
            <span className="bg-navy-950/80 px-3 py-1 rounded-full border border-pulse-green text-pulse-green text-xs font-bold flex items-center gap-2"><span className="w-2 h-2 bg-pulse-green rounded-full shadow-[0_0_8px_rgba(34,197,94,0.8)]"></span>Free</span>
            <span className="bg-navy-950/80 px-3 py-1 rounded-full border border-pulse-yellow text-pulse-yellow text-xs font-bold flex items-center gap-2"><span className="w-2 h-2 bg-pulse-yellow rounded-full shadow-[0_0_8px_rgba(234,179,8,0.8)]"></span>Slow</span>
            <span className="bg-navy-950/80 px-3 py-1 rounded-full border border-pulse-red text-pulse-red text-xs font-bold flex items-center gap-2"><span className="w-2 h-2 bg-pulse-red rounded-full shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span>Jammed</span>
          </div>
          <div className="w-full h-full relative z-0">
             {/* Insert the Map Wrapper here, passing in the live data */}
             <MapWrapper data={networkData || null} />
          </div>
        </div>

        {/* Right: Traffic Nodes List */}
        <div className="bg-navy-900 border border-gray-800 rounded-xl flex flex-col overflow-hidden h-[600px]">
          {/* Header */}
          <div className="p-5 border-b border-gray-800 bg-navy-950/50">
            <h2 className="text-white font-semibold text-lg">Traffic Nodes</h2>
            <p className="text-gray-500 text-xs mt-1">{networkData?.length || 0} monitored intersections</p>
          </div>
          
          {/* Scrollable List */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {networkData?.slice(0, 15).map((node) => (
              <div 
                key={node.id} 
                className="p-4 border-b border-gray-800/50 hover:bg-navy-800/50 transition-colors group"
              >
                {/* Top Row: Name and Badge */}
                <div className="flex justify-between items-start mb-1">
                  <div className="flex items-center gap-3">
                    <span className={`w-2 h-2 rounded-full shadow-[0_0_8px_currentColor] ${
                      node.status === 'Free' ? 'bg-pulse-green text-pulse-green' : 
                      node.status === 'Slow' ? 'bg-pulse-yellow text-pulse-yellow' : 
                      'bg-pulse-red text-pulse-red'
                    }`}></span>
                    <h4 className="text-gray-200 font-medium text-sm group-hover:text-white transition-colors">
                      {node.meta.name}
                    </h4>
                  </div>
                  <span className={`text-[9px] font-bold tracking-wider uppercase px-2 py-0.5 rounded border ${
                    node.status === 'Free' ? 'text-gray-400 border-gray-700 bg-navy-950' : 
                    node.status === 'Slow' ? 'text-pulse-yellow border-pulse-yellow/30 bg-pulse-yellow/10' : 
                    'text-pulse-red border-pulse-red/30 bg-pulse-red/10'
                  }`}>
                    {node.status}
                  </span>
                </div>

                {/* Middle Row: Corridor */}
                <p className="text-[10px] text-gray-500 uppercase tracking-widest ml-5 mb-3">
                  {node.meta.corridor}
                </p>

                {/* Bottom Row: Metrics */}
                <div className="flex justify-between text-xs text-gray-400 ml-5 pr-2">
                  <div className="flex flex-col">
                    <span className="text-gray-300 font-medium">{node.vehicle_count}</span>
                    <span className="text-[10px] text-gray-500">veh</span>
                  </div>
                  <div className="flex flex-col text-center">
                    <span className="text-gray-300 font-medium">{node.speed}</span>
                    <span className="text-[10px] text-gray-500">km/h</span>
                  </div>
                  <div className="flex flex-col text-right">
                    <span className="text-gray-300 font-medium">{Math.round(node.density * 100)}</span>
                    <span className="text-[10px] text-gray-500">v/km</span>
                  </div>
                </div>
              </div>
            ))}

            {(!networkData || networkData.length === 0) && (
              <div className="p-8 text-center text-gray-500 text-sm">
                Waiting for simulator telemetry...
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Charts Section */}
      <DashboardCharts data={networkData || null} />
    </div>
  );
}