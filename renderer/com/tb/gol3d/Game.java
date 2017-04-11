
package com.tb.gol3d;

import static org.lwjgl.glfw.GLFW.GLFW_KEY_ESCAPE;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_LEFT;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_RIGHT;
import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;
import static org.lwjgl.glfw.GLFW.GLFW_RESIZABLE;
import static org.lwjgl.glfw.GLFW.GLFW_VISIBLE;
import static org.lwjgl.glfw.GLFW.glfwCreateWindow;
import static org.lwjgl.glfw.GLFW.glfwDefaultWindowHints;
import static org.lwjgl.glfw.GLFW.glfwDestroyWindow;
import static org.lwjgl.glfw.GLFW.glfwGetPrimaryMonitor;
import static org.lwjgl.glfw.GLFW.glfwGetVideoMode;
import static org.lwjgl.glfw.GLFW.glfwInit;
import static org.lwjgl.glfw.GLFW.glfwMakeContextCurrent;
import static org.lwjgl.glfw.GLFW.glfwPollEvents;
import static org.lwjgl.glfw.GLFW.glfwSetErrorCallback;
import static org.lwjgl.glfw.GLFW.glfwSetKeyCallback;
import static org.lwjgl.glfw.GLFW.glfwSetWindowPos;
import static org.lwjgl.glfw.GLFW.glfwSetWindowShouldClose;
import static org.lwjgl.glfw.GLFW.glfwShowWindow;
import static org.lwjgl.glfw.GLFW.glfwSwapBuffers;
import static org.lwjgl.glfw.GLFW.glfwSwapInterval;
import static org.lwjgl.glfw.GLFW.glfwTerminate;
import static org.lwjgl.glfw.GLFW.glfwWindowHint;
import static org.lwjgl.glfw.GLFW.glfwWindowShouldClose;
import static org.lwjgl.opengl.GL11.GL_COLOR_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.GL_DEPTH_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.GL_DEPTH_TEST;
import static org.lwjgl.opengl.GL11.GL_FALSE;
import static org.lwjgl.opengl.GL11.GL_QUADS;
import static org.lwjgl.opengl.GL11.GL_TRUE;
import static org.lwjgl.opengl.GL11.glBegin;
import static org.lwjgl.opengl.GL11.glClear;
import static org.lwjgl.opengl.GL11.glClearColor;
import static org.lwjgl.opengl.GL11.glColor3f;
import static org.lwjgl.opengl.GL11.glEnable;
import static org.lwjgl.opengl.GL11.glEnd;
import static org.lwjgl.opengl.GL11.glRotatef;
import static org.lwjgl.opengl.GL11.glVertex3f;
import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.system.MemoryUtil.NULL;

import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWKeyCallback;
import org.lwjgl.glfw.GLFWVidMode;
import org.lwjgl.opengl.GL;

/**
 * This class handles rendering a game of life in 3D using LWJGL.
 * 
 * @author Tim Backus tbackus127@gmail.com
 *
 */
public class Game implements Runnable {
  
  /** The width of the render window. */
  private static final int WINDOW_WIDTH = 800;
  
  /** The height of the render window. */
  private static final int WINDOW_HEIGHT = 800;
  
  /** Padding for cube rendering. */
  private static final float AREA_PADDING = 0.75f;
  
  /** The title of the render window. */
  private static final String WINDOW_TITLE = "3D Game of Life Explorer";
  
  /** Handle to the error callback for GLFW. */
  private GLFWErrorCallback errorCallback;
  
  /** Handle to GLFW's key callback. */
  private GLFWKeyCallback keyCallback;
  
  /** Handle to the window. */
  private long window;
  
  /** Window width. */
  private int width = WINDOW_WIDTH;
  
  /** Window height. */
  private int height = WINDOW_HEIGHT;
  
  /** Pointer to game data. */
  private final byte[][][][] gameData;
  
  /** Length of each vertex along the X axis. */
  private final float vertexLengthX;
  
  /** Length of each vertex along the Y axis. */
  private final float vertexLengthY;
  
  /** Length of each vertex along the Z axis. */
  private final float vertexLengthZ;
  
  /** The current iteration number to render. */
  private int itrNumber = 0;
  
  public Game(final byte[][][][] d) {
    this.gameData = d;
    
    // Calculate vertex lengths for all axes
    vertexLengthX = ((float) width / this.gameData[0].length / (float) width) / AREA_PADDING;
    vertexLengthY = ((float) height / this.gameData[0][0].length / (float) height) / AREA_PADDING;
    vertexLengthZ = ((float) 800 / this.gameData[0][0][0].length / 800f) / AREA_PADDING;
    
    System.out.println(vertexLengthX + " " + vertexLengthY + " " + vertexLengthZ);
  }
  
  /**
   * Starts the rendering.
   */
  @Override
  public void run() {
    try {
      init();
      loop();
      glfwDestroyWindow(window);
      keyCallback.free();
    } finally {
      glfwTerminate();
      errorCallback.free();
    }
  }
  
  /**
   * Initializes LWJGL.
   */
  private void init() {
    
    // Tell LWJGL to output its errors to stderr
    glfwSetErrorCallback(errorCallback = GLFWErrorCallback.createPrint(System.err));
    
    if (!glfwInit()) {
      throw new IllegalStateException("Unable to initialize GLFW.");
    }
    
    // Set window hints
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    
    // Create the window
    window = glfwCreateWindow(width, height, WINDOW_TITLE, NULL, NULL);
    if (window == NULL) {
      throw new RuntimeException("Failed to create GLFW window.");
    }
    
    // Capture keystrokes
    glfwSetKeyCallback(window, keyCallback = new GLFWKeyCallback() {
      
      @Override
      public void invoke(long window, int key, int scancode, int action, int mods) {
        
        // Bind ESC to closing the window
        if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
          glfwSetWindowShouldClose(window, true);
        }
        
        // Bind Right Arrow to next GoL iteration
        if (key == GLFW_KEY_RIGHT && action == GLFW_RELEASE) {
          if (itrNumber < gameData.length - 1) {
            itrNumber++;
            System.out.println("Iteration number changed to " + itrNumber);
          }
        }
        
        // Bind Left Arrow to previous GoL iteration
        if (key == GLFW_KEY_LEFT && action == GLFW_RELEASE) {
          if (itrNumber > 0) {
            itrNumber--;
            System.out.println("Iteration number changed to " + itrNumber);
          }
        }
      }
    });
    
    // Get the resolution of the current monitor and center the window
    GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (vidmode.width() - width) >> 1, (vidmode.height() - height) >> 1);
    glfwMakeContextCurrent(window);
    
    // Enable VSync
    glfwSwapInterval(1);
    
    // Show the window
    glfwShowWindow(window);
    
  }
  
  /**
   * Main game loop.
   */
  private void loop() {
    GL.createCapabilities();
    glEnable(GL_DEPTH_TEST);
    
    glClearColor(0.2f, 0.2f, 0.2f, 0.2f);
    
    // Run until the user closes the window
    while (!glfwWindowShouldClose(window)) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
      
      // Do rendering stuff here
      renderCubes();
      
      // Do cube rotation
      glRotatef(0.37f, 1.0f, 0.0f, 0.0f);
      glRotatef(0.17f, 0.0f, 1.0f, 0.0f);
      glRotatef(0.23f, 0.0f, 0.0f, 1.0f);
      
      glfwSwapBuffers(window);
      glfwPollEvents();
    }
  }
  
  /**
   * Renders the cubes for one GoL iteration to the LWGJL window.
   */
  private void renderCubes() {
    glBegin(GL_QUADS);
    
    // Get a pointer to the 3D data for a specific iteration
    final byte[][][] golData = gameData[itrNumber];
    
    // Calculate the position to start drawing cubes
    final float startPos = -1.0f * AREA_PADDING;
    
    // X-coordinates
    for (int i = 0; i < golData.length; i++) {
      
      // Calculate starting and end positions for X-axis
      final float xPos = startPos + vertexLengthX * i;
      final float xEnd = xPos + vertexLengthX;
      
      // Y-coordinates
      for (int j = 0; j < golData[i].length; j++) {
        
        // Calculate start/end positions for Y
        final float yPos = startPos + vertexLengthY * j;
        final float yEnd = yPos + vertexLengthY;
        
        // Z-coordinates
        for (int k = 0; k < golData[i][j].length; k++) {
          
          // Only render the cube if it was alive at this point
          if (golData[i][j][k] > 0) {
            
            // Calculate start/end positions for Z
            final float zPos = startPos + vertexLengthZ * k;
            final float zEnd = zPos + vertexLengthZ;
            
            // Side A of the cube (green)
            glColor3f(0, 1, 0);
            glVertex3f(xEnd, yEnd, zPos);
            glVertex3f(xPos, yEnd, zPos);
            glVertex3f(xPos, yEnd, zEnd);
            glVertex3f(xEnd, yEnd, zEnd);
            
            // Side B (orange)
            glColor3f(1, 0.5f, 0);
            glVertex3f(xEnd, yPos, zEnd);
            glVertex3f(xPos, yPos, zEnd);
            glVertex3f(xPos, yPos, zPos);
            glVertex3f(xEnd, yPos, zPos);
            
            // Side C (red)
            glColor3f(1, 0, 0);
            glVertex3f(xEnd, yEnd, zEnd);
            glVertex3f(xPos, yEnd, zEnd);
            glVertex3f(xPos, yPos, zEnd);
            glVertex3f(xEnd, yPos, zEnd);
            
            // Side D (yellow)
            glColor3f(1, 1, 0);
            glVertex3f(xEnd, yPos, zPos);
            glVertex3f(xPos, yPos, zPos);
            glVertex3f(xPos, yEnd, zPos);
            glVertex3f(xEnd, yEnd, zPos);
            
            // Side E (blue)
            glColor3f(0, 0, 1);
            glVertex3f(xPos, yEnd, zEnd);
            glVertex3f(xPos, yEnd, zPos);
            glVertex3f(xPos, yPos, zPos);
            glVertex3f(xPos, yPos, zEnd);
            
            // Side F (magenta)
            glColor3f(1, 0, 1);
            glVertex3f(xEnd, yEnd, zPos);
            glVertex3f(xEnd, yEnd, zEnd);
            glVertex3f(xEnd, yPos, zEnd);
            glVertex3f(xEnd, yPos, zPos);
          }
        }
      }
    }
    
    // End rendering
    glEnd();
  }
  
}
