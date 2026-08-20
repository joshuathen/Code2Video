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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "PDEs require context to solve uniquely.",
            "Initial conditions define the starting state.",
            "Boundary conditions constrain the edges of space."
        ]
        self.setup_layout("Boundary and Initial Conditions", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        string = Line(start=LEFT*1.5, end=RIGHT*1.5, color="#FFFFFF")
        label = Text("Initial State (t=0)", font_size=22, color="#FFFFFF").scale(0.7)
        initial_state_group = VGroup(string, label).arrange(UP, buff=0.2)
        
        # Fix 1: Apply layout adjustment (Issue 29)
        self.place_in_area(initial_state_group, 'C4', 'D6', scale_factor=0.9)
        
        self.play(Create(string), Write(label))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        new_string = VGroup(
            Line(start=LEFT*1.5, end=ORIGIN, color="#FF00FF"),
            Line(start=ORIGIN, end=RIGHT*1.5, color="#FF00FF")
        )
        new_string[0].rotate(0.5)
        new_string[1].rotate(-0.5)
        new_string.move_to(string.get_center())
        
        self.play(ReplacementTransform(string, new_string))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix 3: Apply boundary lines placement (Issue 31)
        bar1 = Line(UP*0.5, DOWN*0.5, color="#FFFF00", stroke_width=8)
        bar2 = Line(UP*0.5, DOWN*0.5, color="#FFFF00", stroke_width=8)
        boundary_lines = VGroup(bar1, bar2)
        self.place_in_area(boundary_lines, 'C3', 'E4', scale_factor=0.7)
        
        # Fix 2: Apply label placement (Issue 30)
        label_fixed = Text("Fixed Boundaries", font_size=22, color="#FFFF00").scale(0.8)
        self.place_at_grid(label_fixed, 'C5', scale_factor=0.8)
        
        self.play(Create(bar1), Create(bar2), Write(label_fixed))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
