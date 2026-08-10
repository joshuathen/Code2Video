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
        lecture_lines = ["Start with one, two, three points.", "Count the regions for each case.", "Observe the sequence: one, two, four, eight, sixteen."]
        self.setup_layout("Observation: The 'Obvious' Pattern", lecture_lines)
        
        # Assets
        circle_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg"
        
        # Elements
        circle_icon = SVGMobject(circle_svg).set_color("#00FFFF")
        # Ensure circle_icon has geometry before sampling points
        circle_path = Circle(radius=1.5).set_color("#00FFFF")
        dots = [Dot(circle_path.point_from_proportion(i/4)) for i in range(4)]
        dots_group = VGroup(*dots)
        
        # Group geometry
        geometry_group = VGroup(circle_path, dots_group)
        self.place_in_area(geometry_group, 'C2', 'F6', scale_factor=0.9)
        
        # Observation Text
        observation_text = Text("1, 2, 4, 8, 16", font_size=24, color=YELLOW)
        self.place_in_area(observation_text, 'D1', 'F3', scale_factor=0.6)
        
        # SVM Text (as requested by issue)
        svm_definition_text = Text("Sequence Analysis", font_size=20, color=BLUE)
        self.place_in_area(svm_definition_text, 'A1', 'B6', scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(circle_path), Write(dots_group))
        self.lecture[0].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        lines = VGroup(
            Line(dots[0].get_center(), dots[1].get_center(), color=WHITE),
            Line(dots[1].get_center(), dots[2].get_center(), color=WHITE),
            Line(dots[2].get_center(), dots[3].get_center(), color=WHITE),
            Line(dots[3].get_center(), dots[0].get_center(), color=WHITE)
        )
        self.play(Create(lines))
        self.lecture[1].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        diagonals = VGroup(
            Line(dots[0].get_center(), dots[2].get_center(), color="#FF4500"),
            Line(dots[1].get_center(), dots[3].get_center(), color="#FF4500")
        )
        self.play(Create(diagonals), Write(observation_text))
        self.lecture[2].set_color("#FF4500")
        self.wait(1)
