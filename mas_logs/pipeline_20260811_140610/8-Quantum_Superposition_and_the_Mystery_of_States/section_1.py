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
        self.setup_layout("Prerequisite: The Classical Binary", [
            "Classical bits are either '0' or '1'.",
            "A light switch is either 'On' or 'Off'.",
            "Objects exist in definite states daily."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create '0' and '1'
        bit0 = Text("0", font_size=48, color=WHITE)
        bit1 = Text("1", font_size=48, color=WHITE)
        self.place_at_grid(bit0, "B2")
        self.place_at_grid(bit1, "B5")
        self.play(Write(bit0), Write(bit1))
        
        # === Animation for Lecture Line 2 ===
        # Load switch asset (or Square if not found)
        # Using VGroup of a square + circle for simple representation
        switch = VGroup(
            Square(side_length=1.0, color=WHITE),
            Circle(radius=0.3, color=WHITE)
        )
        self.place_at_grid(switch, "D3")
        self.play(FadeIn(switch))
        
        # Toggle color to indicate ON
        self.play(switch.animate.set_color("#FFFF00"))
        self.lecture[1].set_color("#FFFF00")
        
        # === Animation for Lecture Line 3 ===
        # Highlight '0' and '1'
        self.play(
            bit0.animate.set_color("#00FF00"),
            bit1.animate.set_color("#00FF00")
        )
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
