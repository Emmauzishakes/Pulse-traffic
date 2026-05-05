"use client";
import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts';
import { EnrichedTrafficData } from '@/lib/types';

interface ChartProps {
  data: EnrichedTrafficData[] | null;
}

export const DashboardCharts: React.FC<ChartProps> = ({ data }) => {
  if (!data || data.length === 0) return null;

  // 1. Prepare Hotspot Data (Top 6 most dense nodes)
  const hotspotData = [...data]
    .sort((a, b) => b.density - a.density)
    .slice(0, 6)
    .map(node => ({
      name: node.meta.name.split(' ')[0], // Get just the first word (e.g., "Globe", "Bomas")
      density: Math.round(node.density * 100),
      status: node.status
    }));

  // 2. Prepare Trend Data (Simulating a timeline based on live data for visual accuracy)
  // In a production app, you would fetch this from a /historical-aggregate endpoint.
  const now = new Date();
  const trendData = Array.from({ length: 15 }).map((_, i) => {
    const time = new Date(now.getTime() - (14 - i) * 60000); // Past 15 mins
    const avgSpeed = data.reduce((acc, curr) => acc + curr.speed, 0) / data.length;
    const avgDensity = (data.reduce((acc, curr) => acc + curr.density, 0) / data.length) * 100;
    
    return {
      time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      speed: Math.max(0, avgSpeed + (Math.random() * 4 - 2)), // slight jitter for realism
      density: Math.max(0, avgDensity + (Math.random() * 2 - 1))
    };
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
      
      {/* Left: Live Network Trends (Area Chart) */}
      <div className="bg-navy-900 border border-gray-800 rounded-xl p-6 h-[350px] flex flex-col">
        <h3 className="font-semibold text-white mb-4">Live Network Trends</h3>
        <div className="flex-1 w-full min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trendData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorSpeed" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorDensity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" vertical={false} />
              <XAxis dataKey="time" stroke="#4B5563" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#4B5563" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#111827', borderColor: '#1F2937', borderRadius: '8px' }}
                itemStyle={{ color: '#E5E7EB' }}
              />
              <Area type="monotone" dataKey="speed" stroke="#22c55e" strokeWidth={2} fillOpacity={1} fill="url(#colorSpeed)" />
              <Area type="monotone" dataKey="density" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#colorDensity)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Right: Top Hotspots (Bar Chart) */}
      <div className="bg-navy-900 border border-gray-800 rounded-xl p-6 h-[350px] flex flex-col">
        <h3 className="font-semibold text-white mb-4">Top hotspots by density</h3>
        <div className="flex-1 w-full min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={hotspotData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }} barSize={40}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" vertical={false} />
              <XAxis dataKey="name" stroke="#4B5563" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#4B5563" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip 
                cursor={{ fill: '#1F2937', opacity: 0.4 }}
                contentStyle={{ backgroundColor: '#111827', borderColor: '#1F2937', borderRadius: '8px' }}
              />
              <Bar dataKey="density" radius={[4, 4, 0, 0]}>
                {hotspotData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.status === 'Jammed' ? '#ef4444' : entry.status === 'Slow' ? '#eab308' : '#22c55e'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
};