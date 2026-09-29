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
        lecture_lines = [
            "We define midpoint and distance functions.",
            "We seek where these functions vanish.",
            "These functions must hit zero.",
            "Intermediate values force a solution.",
            "Topological maps guarantee our square."
        ]
        self.setup_layout("Topological Mapping", lecture_lines)
        
        # Assets/Mobjects
        # Line 1: Define f(x)
        midpoint_func = MathTex("f(x)", color="#FFFFFF")
        self.place_in_area(midpoint_func, 'A3', 'F5', scale_factor=0.6)
        
        # Line 2: Graph f(x) and f(-x)
        f_x = MathTex("f(x)", color="#00FFFF")
        f_neg_x = MathTex("f(-x)", color="#00FFFF")
        graph_group = VGroup(f_x, f_neg_x).arrange(DOWN)
        self.place_at_grid(graph_group, 'C4', scale_factor=0.5)
        
        # Line 3: Intersection and Asset
        intersection_point = Dot(color="#00FFFF")
        self.place_at_grid(intersection_point, 'B5', scale_factor=0.7)
        anchor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/anchor.svg", color="#00FF00")
        self.place_at_grid(anchor, 'B5', scale_factor=0.4)
        
        # Line 4: Intermediate Values
        ivt_text = Text("IVT guarantees f(c) = 0", font_size=24, color="#FFFF00")
        self.place_at_grid(ivt_text, 'D5', scale_factor=0.6)
        
        # Line 5: Square
        square = Square(side_length=1.0, color="#FF0000")
        self.place_at_grid(square, 'E5', scale_factor=0.5)

        # Animations
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), Write(midpoint_func))
        self.play(self.lecture[1].animate.set_color("#00FFFF"), Write(graph_group))
        self.play(self.lecture[2].animate.set_color("#00FFFF"), FadeIn(intersection_point), FadeIn(anchor))
        self.play(self.lecture[3].animate.set_color("#FFFF00"), Write(ivt_text))
        self.play(self.lecture[4].animate.set_color("#FF0000"), Create(square))
        self.wait(2)
