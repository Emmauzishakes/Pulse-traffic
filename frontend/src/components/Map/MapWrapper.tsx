// "use client";
// import React from 'react';
// import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from 'react-leaflet';
// import 'leaflet/dist/leaflet.css';
// import { EnrichedTrafficData } from '@/lib/types';

// // Coordinates for the Nairobi Corridors
// const JUNCTION_COORDINATES: Record<string, [number, number]> = {
//   "Globe Roundabout (CBD)": [-1.2785, 36.8200],
//   "Museum Hill Interchange": [-1.2755, 36.8145],
//   "Pangani": [-1.2680, 36.8350],
//   "Muthaiga Roundabout": [-1.2600, 36.8400],
//   "Allsopps / Roasters": [-1.2400, 36.8650],
//   "Kasarani Interchange": [-1.2230, 36.8800],
//   "Westlands": [-1.2660, 36.8000],
//   "Kangemi": [-1.2620, 36.7500],
//   "Nyayo Stadium": [-1.3050, 36.8250],
//   "T-Mall Roundabout": [-1.3120, 36.8150],
//   "Prestige Plaza": [-1.2980, 36.7850],
//   "Junction Mall": [-1.2990, 36.7600],
// };

// const getStatusColor = (status: string) => {
//   switch (status?.toLowerCase()) {
//     case 'free': return '#22c55e';   // pulse-green
//     case 'slow': return '#eab308';   // pulse-yellow
//     case 'jammed': return '#ef4444'; // pulse-red
//     default: return '#6b7280';
//   }
// };

// export default function LiveMap({ data }: { data: EnrichedTrafficData[] | null }) {
//   const center: [number, number] = [-1.286389, 36.817223]; 
//   const mapData = data || [];

//   const corridors: Record<string, any[]> = {};
//   mapData.forEach(node => {
//     const corridorName = node.meta.corridor;
//     if (!corridors[corridorName]) corridors[corridorName] = [];
//     corridors[corridorName].push(node);
//   });

//   return (
//     <div className="w-full h-full min-h-[600px] relative z-0">
//       <MapContainer 
//         center={center} 
//         zoom={13} 
//         // Changed className to use absolute sizing to ensure visibility
//         className="absolute inset-0 w-full h-full"
//         style={{ background: '#0a0a0a' }}
//         zoomControl={false}
//       >
//         <TileLayer
//           url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
//           attribution='&copy; CartoDB'
//         />

//         {/* Road Lines (Dashed) */}
//         {Object.entries(corridors).map(([name, nodes]) => {
//           const points = nodes
//             .filter(n => JUNCTION_COORDINATES[n.meta.name])
//             .map(n => JUNCTION_COORDINATES[n.meta.name]);
          
//           return points.length > 1 ? (
//             <Polyline 
//               key={name} 
//               positions={points} 
//               pathOptions={{ 
//                 color: '#38bdf8', 
//                 dashArray: '10, 10', 
//                 weight: 2, 
//                 opacity: 0.4 // Slightly higher opacity for visibility
//               }} 
//             />
//           ) : null;
//         })}

//         {/* Status Nodes */}
//         {mapData.map((node) => {
//           const coords = JUNCTION_COORDINATES[node.meta.name];
//           if (!coords) return null;

//           const color = getStatusColor(node.status);

//           return (
//             <CircleMarker
//               key={node.id}
//               center={coords}
//               radius={9} // Slightly larger for better tap targets
//               pathOptions={{ 
//                 color: color, 
//                 fillColor: color, 
//                 fillOpacity: 0.9, 
//                 weight: 2 
//               }}
//             >
//               <Tooltip direction="top" offset={[0, -10]} opacity={1}>
//                 <div className="text-xs font-sans p-1 bg-white text-gray-900 rounded">
//                   <p className="font-bold border-b border-gray-200 mb-1">{node.meta.name}</p>
//                   <p className="flex justify-between gap-4">
//                     <span>Speed:</span>
//                     <span className="font-semibold">{node.speed} km/h</span>
//                   </p>
//                   <p className="flex justify-between gap-4">
//                     <span>Status:</span>
//                     <span className="font-semibold uppercase" style={{ color }}>{node.status}</span>
//                   </p>
//                 </div>
//               </Tooltip>
//             </CircleMarker>
//           );
//         })}
//       </MapContainer>
//     </div>
//   );
// }






import dynamic from 'next/dynamic';

// Next.js dynamic import ensures Leaflet only renders on the client side
const LiveMap = dynamic(() => import('./LiveMap'), { ssr: false });

export default function MapWrapper({ data, focusedNode }: any) {
  return <LiveMap data={data} focusedNode={focusedNode} />;
}