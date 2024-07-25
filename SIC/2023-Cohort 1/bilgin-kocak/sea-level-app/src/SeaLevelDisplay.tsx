// SeaLevelDisplay.tsx
import React from 'react';
import { Box, Text } from '@chakra-ui/react';

interface SeaLevelDisplayProps {
  seaLevelRise: number;
}

const SeaLevelDisplay: React.FC<SeaLevelDisplayProps> = ({ seaLevelRise }) => {
  return (
    <Box p={4} borderWidth="1px" borderRadius="lg">
      <Text>Year: {seaLevelRise}</Text>
    </Box>
  );
};

export default SeaLevelDisplay;
