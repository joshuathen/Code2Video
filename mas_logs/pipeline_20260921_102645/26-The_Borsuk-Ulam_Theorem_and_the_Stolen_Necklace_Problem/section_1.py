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
        self.setup_layout("Intuitive Prerequisite: The Antipodal Mapping", 
                          ["A sphere has pairs of opposite, antipodal points.", 
                           "Imagine points on the North and South Poles.", 
                           "Antipodal points are exactly on the opposite side."])
        
        # Load asset
        globe_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg"
        globe = SVGMobject(globe_asset)
        
        # Fix: Positioning per VideoCritic (Issue 21, 23)
        self.place_in_area(globe, 'B3', 'E6', scale_factor=0.55)
        self.add(globe)
        
        # Antipodal points (Issue 22)
        # Using Dot as representative for points
        p1 = Dot(color=RED)
        p2 = Dot(color=RED)
        self.place_at_grid(p1, 'A4', scale_factor=1.2)
        self.place_at_grid(p2, 'D4', scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.play(Create(p1), Create(p2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        # Rotate globe with points
        self.play(Rotate(globe, angle=PI/2, axis=RIGHT), 
                  Rotate(p1, angle=PI/2, axis=RIGHT, about_point=globe.get_center()),
                  Rotate(p2, angle=PI/2, axis=RIGHT, about_point=globe.get_center()))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        # Drawing connection
        line = Line(p1.get_center(), p2.get_center(), color=YELLOW, stroke_width=4)
        self.play(Create(line))
        self.wait(2)
