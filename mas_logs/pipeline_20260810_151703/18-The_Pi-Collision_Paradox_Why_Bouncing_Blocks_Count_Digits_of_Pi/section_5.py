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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion and Intuition", [
            "Counting collisions is measuring an angle.",
            "Discrete bounces approximate the circular arc.",
            "Collisions effectively calculate the value of Pi."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        circle = Circle(radius=1.0, color=WHITE)
        self.place_at_grid(circle, 'C4', scale_factor=0.7)
        line = Line(start=circle.get_center(), end=circle.point_at_angle(PI/4), color=YELLOW)
        angle_label = MathTex(r"\\theta", color=YELLOW)
        self.add(circle, line)
        self.place_at_grid(angle_label, 'B4', scale_factor=0.8)
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Discrete bounces arc
        arc = Arc(radius=1.5, angle=PI/3, color=RED, stroke_width=8)
        self.place_at_grid(arc, 'C4', scale_factor=0.7)
        self.lecture[1].set_color(RED)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Pi formula
        pi_formula = MathTex(r"\\text{Number of Collisions} \\sim \\pi \\cdot 10^n", color=WHITE)
        self.place_at_grid(pi_formula, 'E5', scale_factor=1.0)
        self.lecture[2].set_color(WHITE)
        self.play(Write(pi_formula))
        self.wait(2)
