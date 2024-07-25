import React, { useState } from 'react';
import logo from './logo.svg';
import './App.css';
import YearSlider from './YearSlider';
import SeaLevelDisplay from './SeaLevelDisplay';
import SeaLevelAnimation from './SeaLevelAnimation';
import { Box, Text, Container, Stack, HStack, VStack } from '@chakra-ui/react';
import turkeyCities from './turkey_cities.json';

console.log(turkeyCities);

function App() {
  const [selectedYear, setSelectedYear] = useState<number>(2030);
  const [seaLevelRise, setSeaLevelRise] = useState<number>(0);

  const handleYearChange = (year: number) => {
    setSelectedYear(year);
    // Simple linear model for demonstration
    // For example, let's assume the sea level rises 3mm per year
    // const rise = (year - 2000) * 3; // Adjust this model as needed
    const m = 5.340892000000167;
    const n = -3684.5056500003384;
    const baseSeaLevel = 6800;
    const rise = m * year + n - baseSeaLevel;
    console.log(rise);
    setSeaLevelRise(rise / 10);
  };
  const calcLandLossInVietnam = (rise: number) => {
    const landLoss = (rise - 20) * 199.77;

    return landLoss;
  };

  const getAppropriateCity = (rise: number) => {
    const landLoss = calcLandLossInVietnam(rise);
    console.log(landLoss);
    const cities = turkeyCities.filter((city) => {
      return city.Area < landLoss;
    });
    // return cities;
    if (cities.length === 0) {
      return { City: 'No city', Area: 0 };
    }

    // Get the city with the largest area
    const largestCity = cities.reduce((prev, current) => {
      return prev.Area > current.Area ? prev : current;
    });

    return largestCity;
  };

  return (
    <div className="App">
      <Container maxW="container.xl">
        <header className="App-header">
          <HStack spacing="24px">
            <div>
              <Box p={4} borderWidth="1px" borderRadius="lg">
                <Text>
                  Land Loss Approximately in Vietnam{' '}
                  {calcLandLossInVietnam(seaLevelRise).toFixed(2)} km
                  <sup>2</sup>
                </Text>
              </Box>

              <Box p={4} borderWidth="1px" borderRadius="lg">
                <Text>
                  Approximately land {getAppropriateCity(seaLevelRise).City}
                </Text>
              </Box>
              <Box p={4} borderWidth="1px" borderRadius="lg">
                <Text>Rise: {seaLevelRise.toFixed(1)} cm</Text>
              </Box>
              <SeaLevelDisplay seaLevelRise={selectedYear} />
            </div>
            <SeaLevelAnimation seaLevelRise={seaLevelRise} />
          </HStack>

          <YearSlider
            selectedYear={selectedYear}
            onYearChange={handleYearChange}
          />
        </header>
      </Container>
    </div>
  );
}

export default App;
