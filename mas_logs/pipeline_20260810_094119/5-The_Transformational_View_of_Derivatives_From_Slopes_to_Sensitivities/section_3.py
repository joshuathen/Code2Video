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
        self.setup_layout("Mapping Change: The Derivative as a Function-to-Function Operator", [
            "Derivatives map functions to new output functions.",
            "[Asset: Position_Function] transforms via the derivative operator.",
            "[Asset: Velocity_Function] emerges as the new mapped result."
        ])
        
        # Mobjects
        func_box = Rectangle(color=WHITE, height=1.5, width=2.5).add(Text("f(x)", font_size=24))
        op_box = Rectangle(color="#FF0000", height=1.5, width=2.5).add(Text("d/dx", font_size=24))
        result_box = Rectangle(color="#00FF00", height=1.5, width=2.5).add(Text("f'(x)", font_size=24))
        arrow1 = Arrow(start=LEFT, end=RIGHT, color=WHITE, buff=0.1).set_length(1.0)
        arrow2 = Arrow(start=LEFT, end=RIGHT, color=WHITE, buff=0.1).set_length(1.0)
        
        flow = VGroup(func_box, arrow1, op_box, arrow2, result_box).arrange(RIGHT, buff=0.3)
        self.place_in_area(flow, 'C4', 'E6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(func_box))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(arrow1), FadeIn(op_box))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(arrow2), FadeIn(result_box))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
