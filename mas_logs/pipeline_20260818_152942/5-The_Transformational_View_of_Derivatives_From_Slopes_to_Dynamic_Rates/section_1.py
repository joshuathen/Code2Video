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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Secant lines connect two points on a curve.",
            "Average change is calculated as Δy divided by Δx.",
            "A snail moves along a path between two points."
        ]
        self.setup_layout("Prerequisite: The Static Concept (Secant Lines)", lecture_lines)
        
        # Load asset
        snail = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snail.svg")
        self.place_at_grid(snail, 'F3', scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        # Fade in a curve on a #FFFFFF background. (Representing curve as parabola)
        curve = FunctionGraph(lambda x: 0.5 * x**2, x_range=[-2, 2], color=WHITE)
        # Fix: Line 57: self.place_in_area(curve, 'C3', 'E5', scale_factor=0.7)
        self.place_in_area(curve, 'C3', 'E5', scale_factor=0.7)
        self.play(Create(curve), FadeIn(snail))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        p1 = curve.point_from_proportion(0.2)
        p2 = curve.point_from_proportion(0.8)
        secant = Line(p1, p2, color="#FFD700")
        
        point_a = Dot(p1, color="#00BFFF")
        point_b = Dot(p2, color="#00BFFF")
        label_a = MathTex("x", color="#00BFFF", font_size=24).next_to(point_a, DOWN)
        label_b = MathTex("x+h", color="#00BFFF", font_size=24).next_to(point_b, UP)
        
        self.play(Create(secant), FadeIn(point_a), FadeIn(point_b), Write(label_a), Write(label_b))
        
        formula = MathTex(r"m = \frac{\Delta y}{\Delta x}", color="#32CD32")
        # Fix: Line 77: self.place_at_grid(formula, 'A4', scale_factor=1.0)
        self.place_at_grid(formula, "A4", scale_factor=1.0)
        self.play(Write(formula))
        self.lecture[1].set_color("#32CD32")

        # === Animation for Lecture Line 3 ===
        glow = secant.copy().set_stroke(color="#FF4500", width=8, opacity=0.5)
        
        self.play(Create(glow))
        self.play(MoveAlongPath(snail, secant), run_time=2)
        self.lecture[2].set_color("#FF4500")
        
        self.wait(2)
