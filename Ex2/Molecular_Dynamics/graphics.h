#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

GLFWwindow* window;

void initGraphics() {
    glfwInit();
    window = glfwCreateWindow(2100, 2100, "MD Vis", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    glPointSize(10.0f);
}

bool windowShouldClose() {
    return glfwWindowShouldClose(window);
}

void beginFrame() {
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void drawPoint(float x, float y) {
    glBegin(GL_POINTS);
    glVertex2f(x, y);
    glEnd();
}

void endFrame() {
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void cleanupGraphics() {
    glfwTerminate();
}
