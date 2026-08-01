from manim import *
import numpy as np

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

class Section5Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout with section title and lecture points
        lecture_lines = [
            "Consider selecting robot squads in multiples of four.",
            "We plug four roots into our generating function.",
            "The binomial expansion simplifies through polar coordinates quickly.",
            "Complex terms cancel, leaving a single real integer.",
            "The impossible sum is solved with simple complex arithmetic."
        ]
        self.setup_layout("Application: The Robot Selection Protocol", lecture_lines)
        
        robot_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Robot introduces the problem
        robot_intro = SVGMobject(robot_asset_path)
        self.place_at_grid(robot_intro, "A1", scale_factor=0.6)
        
        sum_problem = Text("C(100,0) + C(100,4) + C(100,8) + ...", font_size=24, color=WHITE)
        self.place_in_area(sum_problem, "B1", "B6", scale_factor=0.9)
        
        self.play(FadeIn(robot_intro), Write(sum_problem))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # General Formula and Roots (As per Issue 46)
        # 1. roots_group (A1-D6)
        roots_axes = Axes(x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], x_length=2.5, y_length=2.5, axis_config={"include_tip": False})
        roots_circle = Circle(radius=1.0, color=BLUE_B)
        roots_pts = VGroup(*[Dot(roots_circle.point_at_angle(a), color=YELLOW) for a in [0, PI/2, PI, 3*PI/2]])
        roots_labels = VGroup(
            Text("1", font_size=14).next_to(roots_pts[0], UR, buff=0.1),
            Text("i", font_size=14).next_to(roots_pts[1], UR, buff=0.1),
            Text("-1", font_size=14).next_to(roots_pts[2], UL, buff=0.1),
            Text("-i", font_size=14).next_to(roots_pts[3], DR, buff=0.1)
        )
        roots_group = VGroup(roots_axes, roots_circle, roots_pts, roots_labels)
        self.place_in_area(roots_group, 'A1', 'D6', scale_factor=0.8)
        
        # 2. formula (E3)
        formula = Text("a_r = 1/n Σ ω^{-rj} P(ω^j)", color=YELLOW, font_size=28)
        self.place_at_grid(formula, 'E3', scale_factor=0.7)
        
        # 3. formula_box (F1-F6)
        formula_box = SurroundingRectangle(formula, color=YELLOW, buff=0.2)
        # Note: Critic requested formula at E3 and box at F1-F6.
        self.place_in_area(formula_box, 'F1', 'F6', scale_factor=0.9)

        self.play(FadeOut(robot_intro), FadeOut(sum_problem))
        self.play(Create(roots_axes), Create(roots_circle))
        self.play(FadeIn(roots_pts), Write(roots_labels))
        self.play(Write(formula), Create(formula_box))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Evaluation values
        evals_text = VGroup(
            Text("P(1) = 2^100", font_size=18),
            Text("P(i) = (1+i)^100", font_size=18),
            Text("P(-1) = 0", font_size=18),
            Text("P(-i) = (1-i)^100", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT).set_color("#00FF00")
        self.place_in_area(evals_text, "A4", "D6", scale_factor=1.0)
        
        self.play(Write(evals_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE))
        
        # Simplification via polar coordinates
        polar_text = Text("(1+i)^100 = (sqrt(2)e^{i*pi/4})^100 = -2^50", font_size=20, color=BLUE)
        self.place_in_area(polar_text, "E1", "E6", scale_factor=0.8)
        
        self.play(Write(polar_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        
        # Final result reveal
        final_answer_text = Text("Result = (2^100 - 2^51) / 4", font_size=24, color="#FFD700")
        self.place_at_grid(final_answer_text, "E3", scale_factor=1.0)
        
        # Re-purpose formula_box for the final answer
        final_box = SurroundingRectangle(final_answer_text, color="#FFD700", buff=0.3)
        self.place_in_area(final_box, 'F1', 'F6', scale_factor=1.0)
        
        robot_final = SVGMobject(robot_asset_path)
        self.place_at_grid(robot_final, "F6", scale_factor=0.6)
        
        self.play(
            FadeOut(evals_text), 
            FadeOut(polar_text), 
            FadeOut(formula), 
            FadeOut(formula_box),
            FadeOut(roots_group)
        )
        self.play(Write(final_answer_text), FadeIn(robot_final))
        self.play(Create(final_box))
        self.play(Indicate(final_answer_text, color="#FFD700", scale_factor=1.2))
        self.wait(3)
