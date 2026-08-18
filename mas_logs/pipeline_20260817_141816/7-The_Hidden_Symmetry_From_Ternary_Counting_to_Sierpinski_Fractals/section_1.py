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
        self.setup_layout("The Bridge: From Binary to Ternary Counting", [
            "Base-3 digits represent distinct three-way branch choices.",
            "Each digit dictates one specific path at every junction.",
            "Ternary counting naturally generates recursive, branching structures."
        ])
        
        # Asset loading
        junction_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/junction.svg")
        self.place_at_grid(junction_svg, 'D4', scale_factor=0.5)
        self.add(junction_svg)
        
        # === Animation for Lecture Line 1 ===
        # Display binary digits '0', '1', '2' in light blue (#ADD8E6) centered around junction.svg.
        digits = VGroup(
            Text("0", font_size=36, color="#ADD8E6"),
            Text("1", font_size=36, color="#ADD8E6"),
            Text("2", font_size=36, color="#ADD8E6")
        ).arrange(RIGHT, buff=0.3)
        
        # Align with grid D4 based on critic feedback
        digits.move_to(junction_svg.get_center())
        self.play(Write(digits))
        self.lecture[0].set_color("#ADD8E6")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition '2' to '0' while highlighting the carry bit in yellow (#FFFF00) at junction.svg.
        carry_mark = Circle(radius=0.2, color="#FFFF00").move_to(digits[2].get_center())
        self.play(Create(carry_mark))
        
        new_digit = Text("0", font_size=36, color="#ADD8E6").move_to(digits[2].get_center())
        self.play(Transform(digits[2], new_digit), FadeOut(carry_mark))
        
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Group binary digits to show transition to ternary base in green (#90EE90) originating from junction.svg.
        ternary_group = digits.copy()
        self.play(Indicate(ternary_group, color="#90EE90"))
        
        self.lecture[2].set_color("#90EE90")
        self.wait(2)
