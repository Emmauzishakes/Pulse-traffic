import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  Icon: LucideIcon;
  statusColor?: 'cyan' | 'green' | 'red';
}

export const StatCard: React.FC<StatCardProps> = ({ title, value, subtitle, Icon, statusColor = 'cyan' }) => {
  const colorMap = {
    cyan: 'text-pulse-cyan bg-pulse-cyan/10',
    green: 'text-pulse-green bg-pulse-green/10',
    red: 'text-pulse-red bg-pulse-red/10',
  };

  return (
    <div className="bg-navy-900 border border-gray-800 rounded-xl p-5 flex flex-col justify-between h-full">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-gray-400 text-sm font-medium tracking-wider uppercase">{title}</h3>
        <div className={`p-2 rounded-lg ${colorMap[statusColor]}`}>
          <Icon size={18} />
        </div>
      </div>
      <div>
        <div className="text-4xl font-bold text-white mb-1 flex items-baseline gap-2">
          {value}
        </div>
        <p className="text-gray-500 text-sm">{subtitle}</p>
      </div>
    </div>
  );
};