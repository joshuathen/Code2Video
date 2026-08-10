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
        self.setup_layout("Summary and Geometric Constraint", [
            "If determinant is zero, area collapses.",
            "The vectors v1 and v2 are collinear.",
            "No unique solution exists in this case."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Represent area collapse: a parallelogram flattening into a line
        v1 = Vector([1, 1], color=BLUE)
        v2 = Vector([2, 2], color=RED)
        para = Polygon(ORIGIN, v1.get_end(), v1.get_end() + v2.get_end(), v2.get_end(), color=YELLOW, fill_opacity=0.3)
        
        group = VGroup(v1, v2, para)
        # Applied fix per issues 31, 33
        self.place_in_area(group, 'B3', 'E6', scale_factor=0.45)
        
        self.play(Create(group))
        self.lecture[0].set_color("#FFD700")
        
        # === Animation for Lecture Line 2 ===
        # Highlight collinearity
        self.play(v2.animate.set_color(ORANGE))
        self.lecture[1].set_color(ORANGE)
        
        # === Animation for Lecture Line 3 ===
        # Indicate failure: show a big X or similar
        cross = Cross(color=RED)
        # Applied fix per issues 32, 38
        self.place_at_grid(cross, 'E3', scale_factor=1.0)
        
        self.play(FadeIn(cross))
        self.lecture[2].set_color(RED)
        
        self.play(FadeOut(group), FadeOut(cross))
