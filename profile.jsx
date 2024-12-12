import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { 
  Sparkles, 
  ShieldCheck, 
  TrendingUp, 
  Waves, 
  BookOpen, 
  Briefcase 
} from 'lucide-react';

const SchemeRecommendations = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [userProfile, setUserProfile] = useState(null);

  useEffect(() => {
    // Simulated API call to fetch recommendations
    const fetchRecommendations = async () => {
      try {
        const mockUserProfile = {
          Id: 'UP12345',
          Age: 35,
          Gender: 'Male',
          Income: 75000,
          Occupation: 'Engineer',
          Education: 'Masters'
        };

        const response = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(mockUserProfile)
        });

        const data = await response.json();
        setRecommendations(data.recommendations);
        setUserProfile(mockUserProfile);
      } catch (error) {
        console.error('Error fetching recommendations:', error);
      }
    };

    fetchRecommendations();
  }, []);

  // Color palette inspired by India Post colors
  const COLORS = [
    '#005CAA',  // India Post blue
    '#FFA500',  // Warm orange
    '#228B22',  // Forest green
    '#8B4513',  // Saddle brown
    '#4B0082'   // Indigo
  ];

  // Icons for different scheme aspects
  const SchemeIcons = {
    'Growth Plan': <TrendingUp className="w-6 h-6 text-blue-600" />,
    'Retirement Plan': <ShieldCheck className="w-6 h-6 text-green-600" />,
  };

  const renderProgressBar = (recommendation) => {
    const data = [
      { name: 'Score', value: recommendation.buying_probability_score },
      { name: 'Remaining', value: 100 - recommendation.buying_probability_score }
    ];

    return (
      <div className="flex items-center bg-white shadow-md rounded-lg p-4 mb-4 hover:shadow-xl transition-shadow duration-300">
        <div className="flex-grow">
          <div className="flex items-center mb-2">
            {SchemeIcons[recommendation.scheme_name] || <Sparkles className="w-6 h-6 text-purple-600" />}
            <h3 className="ml-2 text-lg font-semibold text-gray-800">
              {recommendation.scheme_name}
            </h3>
          </div>
          <p className="text-sm text-gray-600">
            Personalized scheme recommendation based on your profile
          </p>
        </div>
        <div className="w-24 h-24">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={40}
                paddingAngle={0}
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={index === 0 ? '#005CAA' : '#E0E0E0'}
                  />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="text-center mt-2 text-sm font-bold text-blue-700">
            {recommendation.buying_probability_score}%
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white shadow-lg rounded-lg p-6 mb-6">
          <div className="flex items-center mb-4">
            <Sparkles className="w-8 h-8 text-purple-600 mr-3" />
            <h1 className="text-2xl font-bold text-gray-800">
              Your Personalized Scheme Recommendations
            </h1>
          </div>
          {userProfile && (
            <div className="bg-blue-50 p-4 rounded-lg mb-4">
              <div className="flex items-center">
                <Briefcase className="w-6 h-6 text-blue-600 mr-3" />
                <p className="text-sm text-gray-700">
                  Profile Analysis: {userProfile.Occupation}, {userProfile.Age} years old
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-4">
          {recommendations.length > 0 ? (
            recommendations.map((recommendation, index) => (
              <div key={index}>
                {renderProgressBar(recommendation)}
              </div>
            ))
          ) : (
            <div className="space-y-4">
              <div className="flex items-center bg-white shadow-md rounded-lg p-4 mb-4">
                <div className="flex-grow">
                  <div className="flex items-center mb-2">
                    <TrendingUp className="w-6 h-6 text-blue-600" />
                    <h3 className="ml-2 text-lg font-semibold text-gray-800">Growth Plan</h3>
                  </div>
                  <p className="text-sm text-gray-600">Loading scheme details...</p>
                </div>
                <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center">
                  <div className="text-sm text-gray-400">--</div>
                </div>
              </div>
              <div className="flex items-center bg-white shadow-md rounded-lg p-4 mb-4">
                <div className="flex-grow">
                  <div className="flex items-center mb-2">
                    <ShieldCheck className="w-6 h-6 text-green-600" />
                    <h3 className="ml-2 text-lg font-semibold text-gray-800">Retirement Plan</h3>
                  </div>
                  <p className="text-sm text-gray-600">Loading scheme details...</p>
                </div>
                <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center">
                  <div className="text-sm text-gray-400">--</div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="mt-6 text-center">
          <div className="bg-blue-100 p-4 rounded-lg inline-block">
            <div className="flex items-center justify-center">
              <BookOpen className="w-6 h-6 text-blue-600 mr-2" />
              <p className="text-sm text-blue-800">
                Powered by Intelligent Recommendation Engine
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SchemeRecommendations;
