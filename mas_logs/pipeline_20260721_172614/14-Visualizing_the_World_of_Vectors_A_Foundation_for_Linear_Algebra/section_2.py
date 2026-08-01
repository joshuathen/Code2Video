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
        # Section title and lecture lines
        title = "The Coordinate System: Vectors as Instructions"
        lines = [
            "- In a grid, vectors act like specific movement instructions.",
            "- The first number tells you how far to move horizontally.",
            "- The second number dictates your vertical steps."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        GRID_COLOR = "#696969"
        VECTOR_COLOR = "#ADFF2F"
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.lecture[0].set_color(GRID_COLOR)
        
        # Create a coordinate grid on the right side
        # Origin at E2 (1.5, -1.8)
        origin_pos = self.grid["E2"]
        
        plane = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": GRID_COLOR,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            },
            axis_config={"include_numbers": False, "color": GRID_COLOR}
        )
        # Shift plane so its (0,0) is at grid coordinate E2
        plane.shift(origin_pos - plane.coords_to_point(0,0))
        
        # Ant at origin [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg]
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg")
        ant.set_color(WHITE)
        ant.scale(0.3)
        ant.move_to(plane.coords_to_point(0,0))
        
        self.play(Create(plane), FadeIn(ant), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlights
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(VECTOR_COLOR)
        
        # Horizontal step of 3 units
        h_line = Line(
            plane.coords_to_point(0, 0),
            plane.coords_to_point(3, 0),
            color=VECTOR_COLOR,
            stroke_width=6
        )
        h_label = MathTex("3", color=VECTOR_COLOR).scale(0.8)
        h_label.next_to(h_line, DOWN, buff=0.1)
        
        self.play(Create(h_line), Write(h_label))
        # Animate "ant" moving horizontally
        self.play(ant.animate.move_to(plane.coords_to_point(3, 0)), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlights
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(VECTOR_COLOR)
        
        # Vertical step of 2 units
        v_line = Line(
            plane.coords_to_point(3, 0),
            plane.coords_to_point(3, 2),
            color=VECTOR_COLOR,
            stroke_width=6
        )
        v_label = MathTex("2", color=VECTOR_COLOR).scale(0.8)
        v_label.next_to(v_line, RIGHT, buff=0.1)
        
        # Resultant vector arrow
        vec = Arrow(
            plane.coords_to_point(0, 0),
            plane.coords_to_point(3, 2),
            buff=0,
            color=VECTOR_COLOR,
            stroke_width=8
        )
        
        # Vector column notation label
        # Fixing Issue 20 & 21: position at C6, scale 0.9
        vec_label = MathTex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=VECTOR_COLOR)
        self.place_at_grid(vec_label, "C6", scale_factor=0.9)

        self.play(Create(v_line), Write(v_label))
        # Animate "ant" moving vertically to the tip
        self.play(ant.animate.move_to(plane.coords_to_point(3, 2)), run_time=1.5)
        self.play(GrowArrow(vec))
        self.play(Write(vec_label))
        self.wait(2)
