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
            "Beyond square roots, i is a master of rotation.",
            "Multiplying by i turns any number ninety degrees left.",
            "It transforms simple algebra into dynamic geometric motion."
        ]
        self.setup_layout("Prerequisite: The Hidden Power of 'i'", lecture_lines)
        
        # Initialize Coordinate System
        plane = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(plane, 'A1', 'F6', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Horizontal white line and point at (1,0)
        horiz_line = Line(plane.c2p(-1.5, 0), plane.c2p(1.5, 0), color=WHITE)
        point_1 = Dot(plane.c2p(1, 0), color=WHITE)
        
        # Symbol 'i' in the top corner (Grid A6)
        symbol_i_corner = Text("i", font_size=48, color=WHITE)
        self.place_at_grid(symbol_i_corner, 'A6')
        
        self.play(
            Create(horiz_line),
            FadeIn(point_1),
            FadeIn(symbol_i_corner),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Vector Asset loading and setup
        vector_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/vector.svg"
        vector = SVGMobject(vector_asset_path)
        vector.set_color(WHITE)
        # Scale to match unit length and align tail to origin
        vector.width = plane.x_axis.get_unit_size()
        vector.move_to(plane.c2p(0, 0), aligned_edge=LEFT)
        
        # Rotation to (0,1)
        label_i = Text("i", color="#ADD8E6").scale(1.2)
        label_i.move_to(plane.c2p(0, 1) + UP * 0.3)
        
        vert_line = Line(plane.c2p(0, -1.5), plane.c2p(0, 1.5), color=WHITE)
        
        self.play(Create(vert_line), FadeIn(vector))
        self.play(
            Rotate(vector, angle=90*DEGREES, about_point=plane.c2p(0, 0)),
            run_time=2,
            rate_func=smooth
        )
        self.play(Write(label_i))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Circular arc showing the path from (1,0) to (-1,0)
        path_arc = Arc(
            radius=plane.x_axis.get_unit_size(),
            start_angle=0,
            angle=PI,
            arc_center=plane.c2p(0, 0),
            color=YELLOW
        )
        
        # Another 90 degree rotation to (-1,0)
        self.play(
            Rotate(vector, angle=90*DEGREES, about_point=plane.c2p(0, 0)),
            Create(path_arc),
            run_time=2,
            rate_func=smooth
        )
        
        # Point at (-1, 0)
        point_minus_1 = Dot(plane.c2p(-1, 0), color=WHITE)
        self.play(FadeIn(point_minus_1))
        
        self.wait(3)
