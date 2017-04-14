  
package com.tb.gol3d.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * This class handles transforming the data from the CUDA 3D Life program into a 4D array that the LWJGL renderer can
 * use.
 * 
 * @author Tim Backus tbackus127@gmail.com
 *
 */
public class WorldBuilder {
  
  /**
   * Builds the 4D array from the passed file name.
   * 
   * @param fileName the relative path to the data file.
   * @return a 4D array with format: [iteration][x-coord][y-coord][z-coord].
   * @throws FileNotFoundException if the file does not exist.
   */
  public static final byte[][][][] buildWorldFromFile(final String fileName) throws FileNotFoundException {
    
    // Set up Scanner
    final File dataFile = new File(fileName);
    final Scanner fileScan = new Scanner(dataFile);
    
    // Begin parsing
    final byte[][][][] result = buildWorld(fileScan);
    
    fileScan.close();
    return result;
  }
  
  /**
   * @param fileScan the Scanner open on the data file.
   * @return a 4D array with format: [iteration][x-coord][y-coord][z-coord].
   */
  private static final byte[][][][] buildWorld(final Scanner fileScan) {
    
    System.out.println("Building world...");
    
    // Get metadata of game
    int iterations = 0;
    int xsize = 0;
    int ysize = 0;
    int zsize = 0;
    if (fileScan.hasNextInt()) iterations = fileScan.nextInt();
    if (fileScan.hasNextInt()) xsize = fileScan.nextInt();
    if (fileScan.hasNextInt()) ysize = fileScan.nextInt();
    if (fileScan.hasNextInt()) zsize = fileScan.nextInt();
    fileScan.nextLine();
    fileScan.nextLine();
    
    // For all iterations in this game
    final byte[][][][] result = new byte[iterations][xsize][ysize][zsize];
    for (int itr = 0; itr < iterations; itr++) {
      System.out.println("Building for iteration " + itr);
      
      // Parse <xsize> lines
      for (int xIndex = 0; xIndex < xsize && fileScan.hasNextLine(); xIndex++) {
        
        final String line = fileScan.nextLine();
        
        // Ensure we're not on the wrong X-Block
        final int xBlockNumber = Integer.parseInt(line.split(":")[0]);
        if (xBlockNumber != xIndex) {
          System.err.println("X-Block ID mismatch! Expected: " + xIndex + ", got " + xBlockNumber + ".");
          return null;
        }
        
        // Split Y-Block and parse
        final String[] yBlockString = line.substring(line.indexOf(':') + 1, line.length()).split(",");
        for (int yIndex = 0; yIndex < ysize; yIndex++) {
          
          // Ensure correct Y-Block size
          if (yIndex >= yBlockString.length) {
            System.err.println("Y-Block length mismatch! Expected " + yIndex + ", got " + yBlockString.length + ".");
            return null;
          }
          
          // Ensure correct Z-Block size
          final String zBlock = yBlockString[yIndex];
          if (zsize != zBlock.length()) {
            System.err.println("Z-Block length mismatch! Expected " + zsize + ", got " + zBlock.length() + ".");
            System.err.println(zBlock);
            return null;
          }
          
          // Extract the individual bits of the Z-Blocks
          for (int zIndex = 0; zIndex < zsize; zIndex++) {
            
            // Populate the data array
            // TODO: Compact the game data (use raw binary instead of ASCII 0's and 1's)
            final byte pop = (zBlock.charAt(zIndex) == '0') ? (byte) 0 : (byte) 1;
            result[itr][xIndex][yIndex][zIndex] = pop;
          }
        }
      }
    }
    
    return result;
  }
}
