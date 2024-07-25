// YearSlider.tsx
import React from 'react';
import {
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
} from '@chakra-ui/react';

interface YearSliderProps {
  selectedYear: number;
  onYearChange: (year: number) => void;
}

const YearSlider: React.FC<YearSliderProps> = ({
  selectedYear,
  onYearChange,
}) => {
  return (
    <Slider
      defaultValue={selectedYear}
      min={2025}
      max={2100}
      step={1}
      onChange={onYearChange}
    >
      <SliderTrack>
        <SliderFilledTrack />
      </SliderTrack>
      <SliderThumb />
    </Slider>
  );
};

export default YearSlider;
