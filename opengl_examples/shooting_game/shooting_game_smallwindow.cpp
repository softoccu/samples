#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm> // For std::remove_if

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

struct Bullet {
    float x, y;
    float speed;
    float r, g, b;
};

struct Bomb {
    float x, y;
    float speed;
};

struct Enemy {
    float x, y;
    bool alive;
    float explosionTime;
    float moveSpeed;
    float moveDirection;
    std::vector<Bomb> bombs;
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void drawRectangle(float x, float y, float width, float height);
void drawTriangle(float x, float y, float base, float height, bool upsideDown = false);
void drawEnemy(const Enemy& enemy);
void drawBullets(const std::vector<Bullet>& bullets);
void drawBomb(const Bomb& bomb);
void drawExplosion(float x, float y, float size);
void drawCannon(float x);
void spawnEnemy(Enemy& enemy, float x);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float cannonWidth = 0.2f;
const float cannonHeight = 0.05f;
const float bulletSpeed = 0.02f;
const float bulletWidth = 0.02f;
const float bulletHeight = 0.05f;
const float enemyWidth = 0.2f; // Increase enemy size to twice
const float enemyHeight = 0.2f; // Increase enemy size to twice
const float bombSpeed = 0.01f;
const float bombWidth = 0.05f;
const float bombHeight = 0.1f;
const float explosionDuration = 1.0f;
const float enemyMoveInterval = 0.5f;

bool cannonAlive = true;
float cannonExplosionTime = 0.0f;

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Shooting Game", NULL, NULL);
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
    float cannonX = 0.0f;
    std::vector<Bullet> bullets;
    std::vector<Bomb> bombs;
    std::vector<Enemy> enemies(5); // Five enemies
    float enemySpacing = (2.0f - 5 * enemyWidth) / 6.0f; // Calculate spacing between enemies

    for (int i = 0; i < 5; ++i) {
        spawnEnemy(enemies[i], -1.0f + enemySpacing + i * (enemyWidth + enemySpacing));
    }

    float lastMoveTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // Move cannon
        if (cannonAlive) {
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS && cannonX - cannonWidth / 2 > -1.0f) {
                cannonX -= 0.02f;
            }
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS && cannonX + cannonWidth / 2 < 1.0f) {
                cannonX += 0.02f;
            }
        }

        // Fire bullets
        if (cannonAlive && (bullets.empty() || bullets.back().y > -0.8f)) {
            bullets.push_back({ cannonX - 0.05f, -1.0f + cannonHeight, bulletSpeed, 1.0f, 0.0f, 0.0f });
            bullets.push_back({ cannonX, -1.0f + cannonHeight, bulletSpeed, 0.0f, 1.0f, 0.0f });
            bullets.push_back({ cannonX + 0.05f, -1.0f + cannonHeight, bulletSpeed, 0.0f, 0.0f, 1.0f });
        }

        // Move bullets
        for (auto& bullet : bullets) {
            bullet.y += bullet.speed;
        }

        // Move enemies
        if (glfwGetTime() - lastMoveTime > enemyMoveInterval) {
            lastMoveTime = glfwGetTime();
            for (auto& enemy : enemies) {
                enemy.x += enemy.moveDirection * enemy.moveSpeed;
                if (enemy.x + enemyWidth / 2 > 1.0f || enemy.x - enemyWidth / 2 < -1.0f) {
                    enemy.moveDirection = -enemy.moveDirection;
                }
                if (rand() % 10 < 3) {
                    Bomb newBomb = { enemy.x, enemy.y - enemyHeight / 2, bombSpeed };
                    bombs.push_back(newBomb);
                    enemy.bombs.push_back(newBomb);
                }
            }
        }

        // Move bombs
        for (auto& bomb : bombs) {
            bomb.y -= bomb.speed;
        }

        // Check for bullet-enemy collisions
        for (auto& bullet : bullets) {
            for (auto& enemy : enemies) {
                if (enemy.alive &&
                    bullet.x < enemy.x + enemyWidth / 2 &&
                    bullet.x + bulletWidth > enemy.x - enemyWidth / 2 &&
                    bullet.y < enemy.y + enemyHeight / 2 &&
                    bullet.y + bulletHeight > enemy.y - enemyHeight / 2) {
                    enemy.alive = false;
                    enemy.explosionTime = glfwGetTime();
                    break;
                }
            }
        }

        // Check for bomb-cannon collisions
        for (auto& bomb : bombs) {
            if (cannonAlive &&
                bomb.x < cannonX + cannonWidth / 2 &&
                bomb.x + bombWidth > cannonX - cannonWidth / 2 &&
                bomb.y < -1.0f + cannonHeight / 2 + cannonHeight &&
                bomb.y + bombHeight > -1.0f + cannonHeight / 2) {
                cannonAlive = false;
                cannonExplosionTime = glfwGetTime();
                break;
            }
        }

        // Remove bombs that are off screen
        bombs.erase(std::remove_if(bombs.begin(), bombs.end(), [](const Bomb& bomb) {
            return bomb.y < -1.0f;
        }), bombs.end());

        // Respawn enemies if exploded
        for (auto& enemy : enemies) {
            if (!enemy.alive && glfwGetTime() - enemy.explosionTime > explosionDuration) {
                spawnEnemy(enemy, enemy.x);
            }
        }

        // Respawn cannon if exploded
        if (!cannonAlive && glfwGetTime() - cannonExplosionTime > explosionDuration) {
            cannonAlive = true;
        }

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Draw cannon
        int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
        if (cannonAlive) {
            glUniform4f(vertexColorLocation, 0.0f, 1.0f, 0.0f, 1.0f);
            drawCannon(cannonX);
        } else {
            glUniform4f(vertexColorLocation, 1.0f, 0.5f, 0.0f, 1.0f);
            drawExplosion(cannonX, -1.0f + cannonHeight / 2, cannonWidth);
        }

        // Draw bullets
        for (const auto& bullet : bullets) {
            glUniform4f(vertexColorLocation, bullet.r, bullet.g, bullet.b, 1.0f);
            drawRectangle(bullet.x, bullet.y, bulletWidth, bulletHeight);
        }

        // Draw enemies
        for (const auto& enemy : enemies) {
            if (enemy.alive) {
                glUniform4f(vertexColorLocation, 1.0f, 0.0f, 0.0f, 1.0f);
                drawEnemy(enemy);
            } else {
                glUniform4f(vertexColorLocation, 1.0f, 0.5f, 0.0f, 1.0f);
                drawExplosion(enemy.x, enemy.y, enemyWidth);
            }
        }

        // Draw bombs
        glUniform4f(vertexColorLocation, 0.5f, 0.5f, 0.5f, 1.0f);
        for (const auto& bomb : bombs) {
            drawBomb(bomb);
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

void drawRectangle(float x, float y, float width, float height) {
    float vertices[] = {
        x - width / 2, y - height / 2, 0.0f,
        x + width / 2, y - height / 2, 0.0f,
        x + width / 2, y + height / 2, 0.0f,
        x - width / 2, y + height / 2, 0.0f
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
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

void drawTriangle(float x, float y, float base, float height, bool upsideDown) {
    float vertices[9];
    if (upsideDown) {
        vertices[0] = x; vertices[1] = y - height / 2; vertices[2] = 0.0f;
        vertices[3] = x - base / 2; vertices[4] = y + height / 2; vertices[5] = 0.0f;
        vertices[6] = x + base / 2; vertices[7] = y + height / 2; vertices[8] = 0.0f;
    } else {
        vertices[0] = x; vertices[1] = y + height / 2; vertices[2] = 0.0f;
        vertices[3] = x - base / 2; vertices[4] = y - height / 2; vertices[5] = 0.0f;
        vertices[6] = x + base / 2; vertices[7] = y - height / 2; vertices[8] = 0.0f;
    }

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void drawEnemy(const Enemy& enemy) {
    drawTriangle(enemy.x, enemy.y, enemyWidth, enemyHeight, true);
}

void drawBullets(const std::vector<Bullet>& bullets) {
    for (const auto& bullet : bullets) {
        drawRectangle(bullet.x, bullet.y, bulletWidth, bulletHeight);
    }
}

void drawBomb(const Bomb& bomb) {
    drawRectangle(bomb.x, bomb.y, bombWidth, bombHeight);
}

void drawExplosion(float x, float y, float size) {
    for (int i = 0; i < 10; ++i) {
        float angle = i * 2.0f * 3.1415926f / 10.0f;
        float dx = cos(angle) * size / 2;
        float dy = sin(angle) * size / 2;
        drawRectangle(x + dx, y + dy, size / 4, size / 4);
    }
}

void drawCannon(float x) {
    drawRectangle(x, -1.0f + cannonHeight / 2, cannonWidth, cannonHeight);

    // Draw three cannon barrels
    drawRectangle(x - 0.05f, -1.0f + cannonHeight + 0.05f, 0.02f, 0.1f);
    drawRectangle(x, -1.0f + cannonHeight + 0.05f, 0.02f, 0.1f);
    drawRectangle(x + 0.05f, -1.0f + cannonHeight + 0.05f, 0.02f, 0.1f);
}

void spawnEnemy(Enemy& enemy, float x) {
    enemy.x = x;
    enemy.y = 1.0f - enemyHeight / 2;
    enemy.alive = true;
    enemy.explosionTime = 0.0f;
    enemy.moveSpeed = 0.02f;
    enemy.moveDirection = (rand() % 2 == 0) ? 1.0f : -1.0f;
    enemy.bombs.clear();
}