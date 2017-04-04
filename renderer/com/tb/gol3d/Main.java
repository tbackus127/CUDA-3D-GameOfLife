
package com.tb.gol3d;

import java.io.FileNotFoundException;
import java.util.Arrays;

import com.tb.gol3d.util.WorldBuilder;

/**
 * Top-level class for 3D Game of Life renderer.
 * 
 * @author Tim Backus tbackus127@gmail.com
 *
 */
@SuppressWarnings("unused")
public class Main {
  
  /**
   * Main method.
   * 
   * @param args runtime args: 0: relative path to the game data file (run the C program first to obtain).
   */
  public static void main(String[] args) {
    
    if (args.length > 0) {
      // Initialize game data
      byte[][][][] data = null;
      
      // Build the world from the passed file
      try {
        data = WorldBuilder.buildWorldFromFile(args[0]);
      } catch (FileNotFoundException e) {
        e.printStackTrace();
      }
    }
    
    // Print out the array to verify it works correctly
    // System.out.println(Arrays.deepToString(data));
    
    // Initialize the blank LWJGL window
    final Game g = new Game();
    g.run();
  }
}
