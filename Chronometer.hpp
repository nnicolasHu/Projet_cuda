// ========================================================================
// ==              Chronometer to measure time elapsed.                  ==
// == Usage :                                                            ==
// == ------                                                             ==
// == Chronometer chrono;                                                ==
// == t1=chrono.click(); // Return time elapsed from last click or start ==
// ========================================================================
#ifndef _CHRONOMETER_HPP_
#define _CHRONOMETER_HPP_

class Chronometer
{
public:
  Chronometer();

  double click();
private:
  double m_time;
};

#endif
