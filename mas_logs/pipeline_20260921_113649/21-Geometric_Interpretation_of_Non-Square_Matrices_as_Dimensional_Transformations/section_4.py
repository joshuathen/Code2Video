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
        self.setup_layout("Visualizing the 'Null Space' Loss", [
            "Null space represents information lost in projection.",
            "Some inputs map directly to zero.",
            "This is the cost of dimensionality reduction."
        ])
        
        # 1. Null space representation
        null_space_rect = Rectangle(width=2, height=1, color="#FFFFFF", fill_opacity=0.3)
        self.place_in_area(null_space_rect, 'A3', 'B4', scale_factor=0.6)
        label_null = Text("Null Space", font_size=20, color="#FFFFFF")
        label_null.next_to(null_space_rect, UP)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(null_space_rect), Write(label_null))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # 2. Animate vectors mapping to zero
        vector = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#FF8080")
        self.place_at_grid(vector, 'E5', scale_factor=0.7)
        target_point = Dot(self.grid["F4"], color="#FF8080")
        
        # === Animation for Lecture Line 2 ===
        self.play(Create(vector))
        self.lecture[1].set_color("#FF8080")
        self.play(vector.animate.move_to(target_point.get_center()), run_time=2)
        self.wait(1)

        # 3. Collapse effect
        circle = Circle(radius=0.5, color="#80FF80")
        self.place_at_grid(circle, 'C5', scale_factor=0.7)
        
        # === Animation for Lecture Line 3 ===
        self.play(Create(circle))
        self.lecture[2].set_color("#80FF80")
        self.play(circle.animate.scale(0.1).set_opacity(0), run_time=2)
        self.wait(1)
