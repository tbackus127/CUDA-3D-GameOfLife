
package com.tb.gol3d;

/**
 * This class contains all data from a game of 3D life.
 * 
 * @author Tim Backus tbackus127@gmail.com
 *
 */
public class World {
  
  /** The game's state at each iteration. */
  // [iteration][x-coord][y-coord][z-coord]
  private final byte[][][][] gameData;
  
  /**
   * Default constructor.
   * 
   * @param data a 4D byte array with game data built from file (via WorldBuilder).
   */
  public World(final byte[][][][] data) {
    this.gameData = data;
  }
  
  /**
   * Gets a cube's data from the game.
   * 
   * @param itr the iteration count.
   * @param x the cube's x position.
   * @param y the cube's y position.
   * @param z the cube's z position.
   * @return a byte of 0 if the cube is dead, 1 if alive.
   */
  public byte get(final int itr, final int x, final int y, final int z) {
    return this.gameData[itr][x][y][z];
  }
}
