#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

GLFWwindow* window;

void initGraphics() {
    glfwInit();
    window = glfwCreateWindow(900, 900, "MD Vis", NULL, NULL);
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

void drawCircle(float x, float y, float radius, int num_segments)
{
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < num_segments; ++i)
    {
        float angle = 2.0f * 3.1415926f * float(i) / float(100);
        float dx = radius * cosf(angle);
        float dy = radius * sinf(angle);
        glVertex2f(x + dx, y + dy);
    }
    glEnd();
}

void drawFilledCircle(float x, float y, float radius, int num_segments)
{
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y); // Center of circle
    for (int i = 0; i <= num_segments; ++i)
    {
        float angle = 2.0f * 3.1415926f * float(i) / float(num_segments);
        float dx = radius * cosf(angle);
        float dy = radius * sinf(angle);
        glVertex2f(x + dx, y + dy);
    }
    glEnd();
}

void endFrame() {
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void cleanupGraphics() {
    glfwTerminate();
}
