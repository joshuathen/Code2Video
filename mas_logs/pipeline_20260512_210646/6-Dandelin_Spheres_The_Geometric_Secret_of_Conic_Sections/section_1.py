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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Mystery of the Slanted Slice"
        lines = [
            'A cone sliced by a plane creates an ellipse.',
            'Is this the same as the 2D definition?',
            'An ellipse has two special points called foci.',
            'The sum of distances to these foci is constant.',
            'How do we prove these 3D and 2D views match?'
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_CONE = "#FF8C00"
        COLOR_PLANE = "#00FFFF"
        COLOR_ELLIPSE = "#FFFFFF"
        COLOR_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_CONE)
        
        # 3D Orange Cone (Using Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cone.svg)
        cone_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/cone.svg"
        cone = SVGMobject(cone_path).set_color(COLOR_CONE).set_fill(COLOR_CONE, opacity=0.3)
        
        # Cyan Slicing Plane (Parallelogram)
        plane = Polygon(
            [-2, 0.5, 0], [2, -0.5, 0], [2.5, -1.2, 0], [-1.5, -0.2, 0],
            color=COLOR_PLANE, fill_opacity=0.4, stroke_width=2
        ).shift(DOWN * 0.2)
        
        visual_group = VGroup(cone, plane)
        # Fix Issue 31: Line 82 (using area B2 to E5)
        self.place_in_area(visual_group, "B2", "E5", scale_factor=0.8)
        
        self.play(FadeIn(cone), run_time=1)
        self.play(FadeIn(plane), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_PLANE)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_ELLIPSE)
        
        # Highlight Ellipse (The intersection)
        ellipse_intersection = Ellipse(width=1.6, height=0.5, color=COLOR_ELLIPSE, stroke_width=4)
        ellipse_intersection.rotate(PI/12)
        ellipse_intersection.move_to(plane.get_center())
        
        ellipse_label = Text("Ellipse", font_size=20, color=COLOR_ELLIPSE)
        # Fix Issue 29: Line 104 (using B6)
        self.place_at_grid(ellipse_label, "B6")
        
        # Foci as Question Marks
        focus1 = Text("?", font_size=24, color=COLOR_PLANE)
        focus2 = Text("?", font_size=24, color=COLOR_PLANE)
        
        focus1.move_to(ellipse_intersection.get_center() + LEFT * 0.4 + UP * 0.05)
        focus2.move_to(ellipse_intersection.get_center() + RIGHT * 0.4 + DOWN * 0.05)
        
        self.play(Create(ellipse_intersection), FadeIn(ellipse_label))
        self.play(FadeIn(focus1), FadeIn(focus2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_ELLIPSE)
        
        def_text = Text("Sum of distances to two foci is constant", font_size=18, color=COLOR_TEXT)
        # Fix Issue 30: Line 122 (using F1 to F6 with scale_factor=0.8)
        self.place_in_area(def_text, "F1", "F6", scale_factor=0.8)
        
        self.play(Write(def_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_PLANE)
        
        # Final emphasis - pulsate the question marks
        self.play(
            focus1.animate.scale(1.2).set_color(WHITE),
            focus2.animate.scale(1.2).set_color(WHITE),
            rate_func=there_and_back
        )
        self.wait(2)
