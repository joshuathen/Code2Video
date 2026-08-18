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
        self.setup_layout("Visualizing the Dual Space", [
            "Visualize dot product results as level sets.",
            "These are parallel lines perpendicular to w.",
            "Vectors on the same line share dot products."
        ])
        
        # Elements
        w_vec = Vector([1, 1], color=YELLOW)
        
        # Grid setup
        lines = VGroup()
        for i in range(-3, 4):
            line = Line(start=[-3, -3 + i, 0], end=[3, 3 + i, 0], color=BLUE, stroke_width=2)
            lines.add(line)
        
        # Apply corrections from Issue #28
        self.place_in_area(lines, "C2", "F5", scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        # Using placeholder asset path as none.svg
        self.play(self.lecture[0].animate.set_color("#F3FF33"))
        self.play(Create(lines))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#F3FF33"))
        # Apply corrections from Issue #27
        self.place_at_grid(w_vec, "A2", scale_factor=0.7)
        self.play(GrowArrow(w_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F3FF33"))
        dot = Dot(color=RED)
        # Apply corrections from Issue #26
        self.place_at_grid(dot, "F5", scale_factor=0.6)
        self.play(FadeIn(dot))
        self.wait(1)
