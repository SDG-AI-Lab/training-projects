// SeaLevelAnimation.tsx
import React from 'react';
import { Box } from '@chakra-ui/react';

interface SeaLevelAnimationProps {
  seaLevelRise: number; // This will control the height of the "water"
}

const SeaLevelAnimation: React.FC<SeaLevelAnimationProps> = ({
  seaLevelRise,
}) => {
  const maxHeight = 300; // Maximum height of the container
  const waterHeight = (seaLevelRise / 100) * maxHeight; // Calculate the height of water

  return (
    <Box
      position="relative"
      width="100%"
      height={`${maxHeight}px`}
      bg="blue.100"
    >
      <Box
        position="absolute"
        bottom="0"
        width="100%"
        height={`${waterHeight}px`}
        bg="blue.500"
        transition="height 0.5s ease-out"
      />
    </Box>
  );
};

export default SeaLevelAnimation;
