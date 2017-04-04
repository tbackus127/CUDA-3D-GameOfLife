
package com.tb.gol3d;

import static org.lwjgl.glfw.GLFW.GLFW_KEY_ESCAPE;
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
import static org.lwjgl.opengl.GL11.GL_FALSE;
import static org.lwjgl.opengl.GL11.GL_TRUE;
import static org.lwjgl.opengl.GL11.glClear;
import static org.lwjgl.opengl.GL11.glClearColor;
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
  private static final int WINDOW_HEIGHT = 600;
  
  /** The title of the render window. */
  private static final String WINDOW_TITLE = "3D Game of Life Explorer";
  
  /** Handle to the error callback for GLFW. */
  private GLFWErrorCallback errorCallback;
  
  @SuppressWarnings("unused")
  /** Handle to GLFW's key callback. */
  private GLFWKeyCallback keyCallback;
  
  /** Handle to the window. */
  private long window;
  
  /**
   * Starts the rendering.
   */
  @Override
  public void run() {
    
    try {
      init();
      loop();
      glfwDestroyWindow(window);
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
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
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
      }
    });
    
    // Get the resolution of the current monitor and center the window
    GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (vidmode.width() - WINDOW_WIDTH) >> 1, (vidmode.height() - WINDOW_HEIGHT) >> 1);
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
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Run until the user closes the window
    while (!glfwWindowShouldClose(window)) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glfwSwapBuffers(window);
      glfwPollEvents();
    }
  }
  
}
