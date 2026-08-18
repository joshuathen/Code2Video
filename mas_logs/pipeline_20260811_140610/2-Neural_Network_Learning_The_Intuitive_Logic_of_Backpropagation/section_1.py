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
        lecture_lines = ["Neural networks aim to minimize prediction errors.", "Think of each weight as a tunable knob.", "Turning these knobs shapes the network output."]
        self.setup_layout("The Learning Objective: The 'Black Box' Problem", lecture_lines)
        
        # Elements
        black_box = Rectangle(width=3, height=2, color=WHITE)
        label = Text("Neural Network", font_size=24, color=WHITE).scale(0.7)
        # Using SVG Asset
        knob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg", color=GREEN)
        input_vec = Arrow(start=LEFT*1, end=RIGHT*1, color=BLUE)
        output_err = Arrow(start=LEFT*1, end=RIGHT*1, color=RED)
        err_label = Text("Error", font_size=20, color=RED)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(black_box, 'B4', 'E6', scale_factor=0.7)
        label.next_to(black_box, UP)
        self.play(Create(black_box), Write(label))
        
        # Position input/output relative to box
        input_vec.next_to(black_box, LEFT)
        output_err.next_to(black_box, RIGHT)
        self.play(FadeIn(input_vec), FadeIn(output_err))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(knob, 'C5', scale_factor=0.5)
        self.play(FadeIn(knob))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        # Critic fix: place_at_grid('error_label', 'C3', scale_factor=0.6)
        self.place_at_grid(err_label, 'C3', scale_factor=0.6)
        self.play(Write(err_label))
        self.play(Rotate(knob, angle=PI/2))
        self.wait(1)
