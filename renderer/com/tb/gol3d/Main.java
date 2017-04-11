
package com.tb.gol3d;

import java.io.FileNotFoundException;

import com.tb.gol3d.util.WorldBuilder;

/**
 * Top-level class for 3D Game of Life renderer.
 * 
 * @author Tim Backus tbackus127@gmail.com
 *
 */
public class Main {
  
  /**
   * Main method.
   * 
   * @param args runtime args: 0: relative path to the game data file (run the C program first to obtain).
   */
  public static void main(String[] args) {
    
    byte[][][][] data = null;
    if (args.length > 0) {
      
      // Build the world from the passed file
      try {
        data = WorldBuilder.buildWorldFromFile(args[0]);
      } catch (FileNotFoundException e) {
        e.printStackTrace();
      }
    } else {
      System.out.println("Usage: java Main <3d_life_data_file.dat");
    }
    
    // In case we returned null for some reason, abort
    if (data == null) {
      System.err.println("Game data not initialized! Aborting.");
      return;
    }
    
    // Initialize the blank LWJGL window
    final Game g = new Game(data);
    g.run();
  }
}
