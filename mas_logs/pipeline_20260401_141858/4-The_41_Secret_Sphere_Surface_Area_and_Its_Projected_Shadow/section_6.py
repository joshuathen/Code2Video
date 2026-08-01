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

class Section6Scene(TeachingScene):
    def construct(self):
        # Section Content and Setup
        self.setup_layout("Summary and Conclusion", [
            "A sphere's shadow is always a circle: π r squared.",
            "The total surface area is always four π r squared.",
            "This 4-to-1 ratio is a fundamental secret of spheres."
        ])
        
        # Initialize lecture lines as dimmed to focus on current line
        for line in self.lecture:
            line.set_color(GREY_E)

        # === Animation for Lecture Line 1 ===
        # Line 1: Shadow is a circle. Match with white circle.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        shadow_circle = Circle(radius=0.7, color="#FFFFFF", stroke_width=4)
        shadow_circle.set_fill("#FFFFFF", opacity=0.3)
        # Position in the left half of the grid
        self.place_in_area(shadow_circle, "B1", "E2", scale_factor=0.85)
        
        self.play(Create(shadow_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Surface area is four circles. Match with grey circles (#808080).
        self.play(self.lecture[1].animate.set_color("#808080"))
        
        # Create a group of four grey circles
        c_template = Circle(radius=0.35, color="#808080", stroke_width=4)
        c_template.set_fill("#808080", opacity=0.3)
        c1, c2, c3, c4 = c_template.copy(), c_template.copy(), c_template.copy(), c_template.copy()
        grey_circles_group = VGroup(c1, c2, c3, c4).arrange_in_grid(rows=2, cols=2, buff=0.25)
        
        # Position in the right half of the grid
        self.place_in_area(grey_circles_group, "B4", "E5", scale_factor=0.9)
        
        self.play(Create(grey_circles_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: The 4-to-1 ratio. Add the colon.
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # White colon placed between the shadow circle and the group of four
        colon = Text(":", color="#FFFFFF", font_size=80)
        self.place_in_area(colon, "C3", "D3", scale_factor=0.5)
        
        self.play(Write(colon))
        self.wait(3)
