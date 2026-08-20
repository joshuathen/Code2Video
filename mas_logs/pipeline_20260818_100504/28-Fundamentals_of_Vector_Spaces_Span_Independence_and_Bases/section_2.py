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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Scale vectors and add them together.",
            "The span is the set of all reach.",
            "Different directions span the whole plane."
        ]
        self.setup_layout("Linear Combinations & Span", lecture_lines)
        
        # Initialize Mobjects
        v1 = Arrow(ORIGIN, RIGHT, color="#00FF00")
        v2 = Arrow(ORIGIN, UP, color="#00FF00")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.place_at_grid(v1, 'B2', scale_factor=0.7)
        self.place_at_grid(v2, 'B3', scale_factor=0.7)
        self.play(Create(v1), Create(v2))
        
        # Manually construct resulting vector for addition
        v_sum = Vector(v1.get_end() + v2.get_end(), color="#FFFF00")
        self.play(ReplacementTransform(VGroup(v1.copy(), v2.copy()), v_sum))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        span_rect = Rectangle(width=2, height=2, color="#00FF00", fill_opacity=0.3)
        self.place_in_area(span_rect, 'D3', 'E4', scale_factor=0.7)
        self.play(FadeIn(span_rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        plane = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], background_line_style={"stroke_opacity": 0.5}).scale(0.5)
        self.place_in_area(plane, 'D3', 'F5', scale_factor=0.9)
        self.play(FadeIn(plane))
        self.wait(2)
