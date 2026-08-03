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

class Section6Scene(TeachingScene):
    def construct(self):
        # Data
        title = "Application: Why Abstraction Matters"
        lines = [
            "Abstraction lets us treat images as high-dimensional points.",
            "Facial recognition calculates distances between these data vectors.",
            "Linear algebra solves complex problems across many fields."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_IMAGE = "#ADD8E6"  # Light blue
        COLOR_BASIS = "#D3D3D3"  # Light grey (from storyboard)
        COLOR_POINT = "#FF69B4"  # Hot pink
        COLOR_DISTANCE = "#FFFF00"  # Yellow (from storyboard)
        
        # === Animation for Lecture Line 1 ===
        # Line: "Abstraction lets us treat images as high-dimensional points."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Load asset for face
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/face.svg
        face_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/face.svg", color=COLOR_IMAGE)
        self.place_in_area(face_svg, "B3", "C4", scale_factor=1.2)
        
        self.play(DrawBorderThenFill(face_svg))
        self.wait(1)
        
        # Basis images (smaller grids representing components)
        basis_vgroup = VGroup()
        for i in range(3):
            b = VGroup(*[Square(side_length=0.1, fill_opacity=0.3, color=COLOR_BASIS, stroke_width=1) for _ in range(9)]).arrange_in_grid(rows=3, cols=3, buff=0.02)
            basis_vgroup.add(b)
        
        # Issue 34: Move to row E to reduce gap
        self.place_at_grid(basis_vgroup[0], "E2")
        self.place_at_grid(basis_vgroup[1], "E4")
        self.place_at_grid(basis_vgroup[2], "E6")
        
        plus1 = Text("+", font_size=24, color=WHITE).move_to((self.grid["E2"] + self.grid["E4"])/2)
        plus2 = Text("+", font_size=24, color=WHITE).move_to((self.grid["E4"] + self.grid["E6"])/2)
        
        # Arrow from below the main face to the basis area
        arrow = Arrow(self.grid["D4"], self.grid["E4"], buff=0.3, color=WHITE, stroke_width=3)
        
        self.play(
            Create(arrow),
            FadeIn(basis_vgroup, shift=UP),
            Write(plus1),
            Write(plus2)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Line: "Facial recognition calculates distances between these data vectors."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            FadeOut(face_svg, basis_vgroup, plus1, plus2, arrow)
        )
        
        # High-dimensional coordinate system (Axes)
        # Issue 35: Reduce vertical span and scale
        axes = Axes(
            x_range=[0, 5], y_range=[0, 5], 
            axis_config={"include_tip": True, "stroke_width": 2},
            x_length=3.5, y_length=2.5
        )
        self.place_in_area(axes, "B2", "E6", scale_factor=0.8)
        
        p1_coord = axes.c2p(1, 4)
        p2_coord = axes.c2p(4, 1.5)
        
        dot1 = Dot(p1_coord, color=COLOR_POINT)
        dot2 = Dot(p2_coord, color=COLOR_POINT)
        label1 = Text("Face A", font_size=16).next_to(dot1, UP, buff=0.1)
        label2 = Text("Face B", font_size=16).next_to(dot2, RIGHT, buff=0.1)
        
        self.play(Create(axes))
        self.play(
            FadeIn(dot1, dot2, scale=0.5),
            Write(label1),
            Write(label2)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line: "Linear algebra solves complex problems across many fields."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Measuring line and distance label
        dist_line = Line(p1_coord, p2_coord, color=COLOR_DISTANCE, stroke_width=4)
        dist_label = Text("Distance", font_size=20, color=COLOR_DISTANCE).next_to(dist_line.get_center(), UR, buff=0.1)
        
        self.play(Create(dist_line))
        self.play(Write(dist_label))
        self.wait(3)
        
        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
