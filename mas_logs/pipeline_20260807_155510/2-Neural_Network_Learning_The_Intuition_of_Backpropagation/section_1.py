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
            "Neural networks are machines that learn from mistakes.",
            "Imagine a student shooting a basketball at a hoop.",
            "The distance from the hoop is the error."
        ]
        self.setup_layout("The Analogy: Learning through Trial and Error", lecture_lines)
        
        # Elements
        # Asset integration (Issue 16)
        student = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/student.svg")
        hoop = Rectangle(height=0.3, width=1.0, color=WHITE)
        ball = Dot(color=ORANGE)
        
        # Issue 21: Positioning fix
        self.place_at_grid(student, 'B2', scale_factor=0.7)
        self.place_at_grid(hoop, 'B4', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        # Show a student solving a math problem by trial.
        self.play(FadeIn(student))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imagine a student shooting a basketball at a hoop.
        self.play(FadeIn(hoop), FadeIn(ball))
        ball.move_to(student.get_center())
        self.play(ball.animate.move_to(hoop.get_center() + DOWN*0.2), run_time=1.5)
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The distance from the hoop is the error.
        error_line = DashedLine(ball.get_center(), hoop.get_center(), color=GREEN)
        self.play(Create(error_line))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
