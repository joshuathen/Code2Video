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
        self.setup_layout("Breaking the Pattern: The Geometric Reality", [
            "Add the 6th point carefully.",
            "The pattern breaks at 31.",
            "Visual intersection reality defies expectation."
        ])

        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg").scale(0.3)
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg").scale(0.3)

        circle = Circle(radius=1.5, color=WHITE)
        self.place_at_grid(circle, 'C5', scale_factor=0.6)
        
        points = [
            Dot(color="#00CED1").move_to(circle.point_at_angle(angle))
            for angle in np.linspace(0, 2*PI, 7)[:-1]
        ]
        points_group = VGroup(*points)

        edges = VGroup()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                edges.add(Line(points[i].get_center(), points[j].get_center(), color="#FFFF00", stroke_width=2))
        
        shape_group = VGroup(circle, points_group, edges)
        self.place_at_grid(shape_group, 'D4', scale_factor=0.7)
        
        grid_title = Text("Geometric Network", font_size=20, color=BLUE)
        self.place_in_area(grid_title, 'A4', 'A6', scale_factor=0.5)

        # === Animation for Lecture Line 1: Add the 6th point carefully. ===
        self.lecture[0].set_color("#00CED1")
        self.place_at_grid(compass, 'A1', scale_factor=0.5)
        self.play(FadeIn(compass), Create(circle), FadeIn(points_group))

        # === Animation for Lecture Line 2: The pattern breaks at 31. ===
        self.lecture[1].set_color("#FFFF00")
        self.play(Create(edges))

        # === Animation for Lecture Line 3: Visual intersection reality defies expectation. ===
        self.lecture[2].set_color("#ADFF2F")
        self.place_at_grid(protractor, 'F1', scale_factor=0.5)
        self.play(FadeIn(protractor), Rotate(shape_group, angle=PI/4, run_time=2))
        self.play(Flash(shape_group, color="#ADFF2F", line_length=0.2))
