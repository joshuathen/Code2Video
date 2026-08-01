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

class Section4Scene(Scene):
    def construct(self):
        # 1. Setup Standard Coordinate System
        plane = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        )
        self.play(Create(plane))
        
        # 2. Define a vector in the standard basis
        vector_v = Vector([2, 1], color=YELLOW)
        # Using Text instead of MathTex to avoid FileNotFoundError for 'latex'
        label_v = Text("v = [2, 1]", color=YELLOW).scale(0.6)
        label_v.next_to(vector_v.get_end(), UR, buff=0.1)
        
        self.play(GrowArrow(vector_v), Write(label_v))
        self.wait(1)

        # 3. Introduce a new basis (B)
        b1 = Vector([1, 1], color=PINK)
        b2 = Vector([-1, 1], color=ORANGE)
        
        # Using Text instead of MathTex for basis labels
        label_b1 = Text("b1", color=PINK).scale(0.6).next_to(b1.get_end(), RIGHT)
        label_b2 = Text("b2", color=ORANGE).scale(0.6).next_to(b2.get_end(), LEFT)
        
        basis_group = VGroup(b1, b2, label_b1, label_b2)
        
        self.play(
            Create(b1),
            Create(b2),
            Write(label_b1),
            Write(label_b2)
        )
        self.wait(2)
