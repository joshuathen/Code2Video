from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Step-by-Step Procedure: The Robotic Workflow", [
            "Differentiate both sides by x.",
            "Use Chain Rule on y terms.",
            "Group all dy/dx terms together.",
            "Factor out dy/dx to solve.",
            "The derivative is now isolated."
        ])
        
        # Load assets
        robot1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Animations
        eq1 = MathTex(r"\frac{d}{dx}(x^2 + y^2) = \frac{d}{dx}(25)")
        step2 = MathTex(r"2x + 2y \frac{dy}{dx} = 0")
        step3 = MathTex(r"2y \frac{dy}{dx} = -2x")
        step4 = MathTex(r"\frac{dy}{dx} = -\frac{x}{y}")
        
        # Group equations
        equation_group = VGroup(eq1, step2, step3, step4).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(robot1, 'A1', scale_factor=0.3)
        self.place_in_area(equation_group, 'B4', 'E6', scale_factor=0.9)
        self.play(Write(eq1), FadeIn(robot1))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(Write(step2))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(Write(step3))
        self.lecture[2].set_color("#00FFFF")
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.play(Write(step4))
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.place_at_grid(robot2, 'F6', scale_factor=0.3)
        self.play(FadeIn(robot2))
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
