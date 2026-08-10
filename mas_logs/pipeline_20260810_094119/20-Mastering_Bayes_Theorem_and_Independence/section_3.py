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
        lecture_lines = [
            "Bayes' theorem bridges P(A|B) and P(B|A).",
            "It updates our prior beliefs using new evidence.",
            "P(A|B) = [P(B|A) * P(A)] / P(B).",
            "Imagine a robot updating its hunt for a target.",
            "New rustling (evidence) changes the fox probability."
        ]
        self.setup_layout("Introducing Bayes' Theorem: Updating Beliefs", lecture_lines)
        
        # Colors for lecture lines
        line_colors = ["#FFD700", "#7FFF00", "#FF00FF", "#00BFFF", "#FF4500"]

        # Equation components
        bayes_eq = MathTex("P(H|E) = \\frac{P(E|H) \\cdot P(H)}{P(E)}", font_size=40)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(line_colors[0]))
        self.place_in_area(bayes_eq, 'A2', 'B5', scale_factor=0.9)
        self.play(Write(bayes_eq))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(line_colors[1]))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(line_colors[2]))
        # Highlight components P(H|E) and P(E|H)
        label_h_e = MathTex("P(H|E)", color="#00FFFF")
        label_e_h = MathTex("P(E|H)", color="#00FFFF")
        self.place_at_grid(label_h_e, 'C2', scale_factor=0.9)
        self.place_at_grid(label_e_h, 'C4', scale_factor=0.9)
        self.play(FadeIn(label_h_e), FadeIn(label_e_h))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(line_colors[3]))
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'E2', scale_factor=0.8)
        self.play(FadeIn(robot))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(line_colors[4]))
        norm_const = MathTex("P(E)", color="#FF0000")
        self.place_at_grid(norm_const, 'E4', scale_factor=0.8)
        fox = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fox.svg")
        self.place_at_grid(fox, 'E6', scale_factor=0.8)
        self.play(Write(norm_const), FadeIn(fox))
        self.play(Indicate(norm_const))
        self.wait(2)
