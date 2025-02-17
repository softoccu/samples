#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

const char* vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos, 1.0);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core
out vec4 FragColor;
uniform vec4 ourColor;

void main()
{
    FragColor = ourColor;
}
)glsl";

struct Ball {
    float x, y;
    float radius;
    float xSpeed, ySpeed;
    float r, g, b;
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void drawRectangle();
void drawCircle(float cx, float cy, float r, int num_segments);
void handleCollisions(Ball& ball1, Ball& ball2);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const int numBalls = 3;

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Bouncing Balls", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    srand(static_cast<unsigned int>(time(0)));
    Ball balls[numBalls];
    for (int i = 0; i < numBalls; ++i) {
        balls[i].x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1.5f - 0.75f;
        balls[i].y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1.5f - 0.75f;
        balls[i].radius = 0.05f;
        balls[i].xSpeed = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.02f) - 0.01f;
        balls[i].ySpeed = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.02f) - 0.01f;
        balls[i].r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        balls[i].g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        balls[i].b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        for (int i = 0; i < numBalls; ++i) {
            balls[i].x += balls[i].xSpeed;
            balls[i].y += balls[i].ySpeed;

            if (balls[i].x + balls[i].radius > 1.0f || balls[i].x - balls[i].radius < -1.0f) {
                balls[i].xSpeed = -balls[i].xSpeed;
                balls[i].r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                balls[i].g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                balls[i].b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }

            if (balls[i].y + balls[i].radius > 0.75f || balls[i].y - balls[i].radius < -0.75f) {
                balls[i].ySpeed = -balls[i].ySpeed;
                balls[i].r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                balls[i].g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                balls[i].b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }

            for (int j = i + 1; j < numBalls; ++j) {
                handleCollisions(balls[i], balls[j]);
            }
        }

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        drawRectangle();

        for (int i = 0; i < numBalls; ++i) {
            int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
            glUniform4f(vertexColorLocation, balls[i].r, balls[i].g, balls[i].b, 1.0f);
            drawCircle(balls[i].x, balls[i].y, balls[i].radius, 100);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

void drawRectangle() {
    float vertices[] = {
        -1.0f, -0.75f, 0.0f,  
         1.0f, -0.75f, 0.0f,  
         1.0f,  0.75f, 0.0f,  
        -1.0f,  0.75f, 0.0f  
    };

    unsigned int indices[] = {
        0, 1, 2,  
        2, 3, 0   
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindVertexArray(VAO);
    glDrawElements(GL_LINE_LOOP, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

void drawCircle(float cx, float cy, float r, int num_segments) {
    float theta = 2 * 3.1415926f / float(num_segments);
    float c = cosf(theta);
    float s = sinf(theta);
    float t;

    float x = r;
    float y = 0;

    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(cx, cy);
    for (int i = 0; i <= num_segments; i++) {
        glVertex2f(x + cx, y + cy);
        t = x;
        x = c * x - s * y;
        y = s * t + c * y;
    }
    glEnd();
}

void handleCollisions(Ball& ball1, Ball& ball2) {
    float dx = ball2.x - ball1.x;
    float dy = ball2.y - ball1.y;
    float distance = sqrt(dx * dx + dy * dy);

    if (distance < ball1.radius + ball2.radius) {
        float angle = atan2(dy, dx);
        float targetX = ball1.x + cos(angle) * (ball1.radius + ball2.radius);
        float targetY = ball1.y + sin(angle) * (ball1.radius + ball2.radius);
        float ax = (targetX - ball2.x) * 0.5f;
        float ay = (targetY - ball2.y) * 0.5f;

        ball1.x -= ax;
        ball1.y -= ay;
        ball2.x += ax;
        ball2.y += ay;

        float angle1 = atan2(ball1.ySpeed, ball1.xSpeed);
        float angle2 = atan2(ball2.ySpeed, ball2.xSpeed);

        float speed1 = sqrt(ball1.xSpeed * ball1.xSpeed + ball1.ySpeed * ball1.ySpeed);
        float speed2 = sqrt(ball2.xSpeed * ball2.xSpeed + ball2.ySpeed * ball2.ySpeed);

        float dir1 = angle1 - angle;
        float dir2 = angle2 - angle;

        float new_xSpeed1 = speed1 * cos(dir1);
        float new_ySpeed1 = speed1 * sin(dir1);
        float new_xSpeed2 = speed2 * cos(dir2);
        float new_ySpeed2 = speed2 * sin(dir2);

        ball1.xSpeed = cos(angle) * new_xSpeed2 + cos(angle + 3.1415926f / 2.0f) * new_ySpeed1;
        ball1.ySpeed = sin(angle) * new_xSpeed2 + sin(angle + 3.1415926f / 2.0f) * new_ySpeed1;
        ball2.xSpeed = cos(angle) * new_xSpeed1 + cos(angle + 3.1415926f / 2.0f) * new_ySpeed2;
        ball2.ySpeed = sin(angle) * new_xSpeed1 + sin(angle + 3.1415926f / 2.0f) * new_ySpeed2;

        ball1.r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ball1.g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ball1.b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        ball2.r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ball2.g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ball2.b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}
