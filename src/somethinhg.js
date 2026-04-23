/**
 * Doom Slayer - Roast Messages Collection
 * Motivational (harsh) messages to display when doomscrolling is detected
 */
// asd
export const roasts = [
  "You'll fail if you don't stop!",
  "Your dreams called - they want your attention back!",
  "Scrolling won't make that deadline disappear!",
  "The phone can wait. Your future can't." ,
  "Success doesn't scroll itself into existence!",
  "That screen won't study for you!",
  "Your goals > Your feed. Remember that.",
  "Future you is watching. They're disappointed.",
  "Every scroll is a step backward. Look up!",
  "The algorithm wins again. Pathetic.",
  "Is this really more important than your goals?",
  "Your productivity just left the chat.",
  "Doomscrolling detected! You're better than this!",
  "PUT. THE. PHONE. DOWN. NOW.",
  "This is why you're behind schedule.",
  "Your potential is crying right now.",
  "Champions don't doomscroll. Be a champion.",
  "The only thing you're winning is wasted time.",
  "Your phone doesn't love you back.",
  "Tick tock. That's your life slipping away.",
];

/**
 * Get a random roast message
 * @returns {string} A random roast message
 */ 
export function  getRandomRoast() {
  return roasts[Math.floor(Math.random() * roasts.length)];
}

/**
 * Get positive messages for good posture                                                      
 */
export const encouragements = [
  "Good posture! Keep it up! 💪",
  "That's the focus we love to see!",
  "You're crushing it! Eyes on the prize!",
  "Productivity mode: ACTIVATED",
  "This is how winners operate!",
];

/**
 * Get a random encouragement message
 * @returns {string} A random encouragement message
 */
export function getRandomEncouragement() {
  return encouragements[Math.floor(Math.random() * encouragements.length)];
}
