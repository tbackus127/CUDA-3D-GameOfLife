
package com.tb.gol3d.util;

/**
 * This class simply acts as a 3-float tuple.
 * 
 * @author Tim Backus tbackus127@gmail.com
 *
 */
public class Vector3f {
  
  /** X component. */
  private final float x;
  
  /** Y component. */
  private final float y;
  
  /** Z component. */
  private final float z;
  
  /**
   * Default constructor. Constructs a new Vector3f with values (0, 0, 0).
   */
  public Vector3f() {
    this.x = 0;
    this.y = 0;
    this.z = 0;
  }
  
  /**
   * Constructs a new Vector3f with the specified values.
   * 
   * @param x the X-component.
   * @param y the Y-component.
   * @param z the Z-component.
   */
  public Vector3f(final float x, final float y, final float z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  
  /**
   * Gets the X component.
   * 
   * @return the X component as a float.
   */
  public float getX() {
    return x;
  }
  
  /**
   * Gets the Y component.
   * 
   * @return the Y component as a float.
   */
  public float getY() {
    return y;
  }
  
  /**
   * Gets the Z component.
   * 
   * @return the Z component as a float.
   */
  public float getZ() {
    return z;
  }
}
