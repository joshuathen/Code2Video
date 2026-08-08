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
        self.setup_layout("Defining the n-Sphere", [
            "Sⁿ is the surface; Bⁿ is the solid ball.",
            "An n-sphere lives in n+1 dimensional space.",
            "Each new dimension adds a new axis of symmetry."
        ])
        
        # Colors
        s_color = "#FFFFFF"
        b_color = "#808080"
        highlight_color = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # "Sⁿ is the surface; Bⁿ is the solid ball."
        self.play(self.lecture[0].animate.set_color(highlight_color))
        
        # S0: Two points
        s0_p1 = Dot(radius=0.08, color=s_color)
        s0_p2 = Dot(radius=0.08, color=s_color)
        s0 = VGroup(s0_p1, s0_p2).arrange(RIGHT, buff=0.6)
        self.place_at_grid(s0, "B2")
        s0_label = MathTex("S^0", color=s_color)
        self.place_at_grid(s0_label, "C2", scale_factor=0.8)
        
        # S1: Circle
        s1 = Circle(radius=0.4, color=s_color)
        self.place_at_grid(s1, "B4")
        s1_label = MathTex("S^1", color=s_color)
        self.place_at_grid(s1_label, "C4", scale_factor=0.8)
        
        # S2: Sphere (2D representation)
        s2_outline = Circle(radius=0.4, color=s_color)
        s2_equator = Ellipse(width=0.8, height=0.2, color=s_color).set_stroke(opacity=0.5)
        s2 = VGroup(s2_outline, s2_equator)
        self.place_at_grid(s2, "B6")
        s2_label = MathTex("S^2", color=s_color)
        self.place_at_grid(s2_label, "C6", scale_factor=0.8)
        
        self.play(
            FadeIn(s0), Write(s0_label),
            Create(s1), Write(s1_label),
            Create(s2), Write(s2_label)
        )
        self.wait(1)
        
        # B1 (disk) - Interior of S1 (using storyboard notation)
        b1 = Circle(radius=0.4, fill_opacity=0.6, fill_color=b_color, stroke_width=0)
        self.place_at_grid(b1, "B4")
        b1_label = MathTex("B^1", color=b_color)
        self.place_at_grid(b1_label, "D4", scale_factor=0.8)

        # B2 (solid ball) - Interior of S2
        b2 = Circle(radius=0.4, fill_opacity=0.6, fill_color=b_color, stroke_width=0)
        self.place_at_grid(b2, "B6")
        b2_label = MathTex("B^2", color=b_color)
        self.place_at_grid(b2_label, "D6", scale_factor=0.8)

        self.play(
            FadeIn(b1), Write(b1_label),
            FadeIn(b2), Write(b2_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # "An n-sphere lives in n+1 dimensional space."
        line2_color = BLUE
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(line2_color)
        )
        
        # Relationship Table
        table_title = Text("Mapping n-Sphere to Space", font_size=22, color=line2_color)
        self.place_in_area(table_title, "E2", "E5", scale_factor=0.8)
        
        table_data = [
            ["n", "n-Sphere", "Space Dim"],
            ["0", "Two Points", "1D Line"],
            ["1", "Circle", "2D Plane"],
            ["2", "Hollow Globe", "3D Space"]
        ]
        
        table_vgroup = VGroup()
        for i, row in enumerate(table_data):
            row_vgroup = VGroup(*[Text(cell, font_size=18, color=WHITE if i == 0 else LIGHT_GRAY) for cell in row])
            row_vgroup.arrange(RIGHT, buff=0.5)
            table_vgroup.add(row_vgroup)
        table_vgroup.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        
        self.place_in_area(table_vgroup, "F1", "F6", scale_factor=0.8)
        
        self.play(Write(table_title))
        self.play(FadeIn(table_vgroup))
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        # "Each new dimension adds a new axis of symmetry."
        line3_color = GREEN
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(line3_color)
        )
        
        # Highlight axes of symmetry
        # S1 axes (x and y)
        s1_axis_x = Line(s1.get_left(), s1.get_right(), color=line3_color, stroke_width=2)
        s1_axis_y = Line(s1.get_top(), s1.get_bottom(), color=line3_color, stroke_width=2)
        
        # S2 axis (vertical)
        s2_axis_z = Line(s2.get_top(), s2.get_bottom(), color=line3_color, stroke_width=2)
        
        # Rotation arrow to indicate symmetry
        rot_arrow = Arc(radius=0.2, start_angle=0, angle=PI*1.5, color=line3_color).add_tip()
        self.place_at_grid(rot_arrow, "A4", scale_factor=0.5)
        
        self.play(Create(s1_axis_x), Create(s1_axis_y))
        self.play(Create(s2_axis_z), GrowArrow(rot_arrow))
        
        self.wait(3)
