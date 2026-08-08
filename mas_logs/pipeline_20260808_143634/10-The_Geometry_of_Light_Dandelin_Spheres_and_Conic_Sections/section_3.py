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
            "Points on the curve satisfy a distance property.",
            "Distances to tangency points are constant for ellipses.",
            "This creates the definition of an ellipse.",
            "The sphere tangency points act as the foci.",
            "Geometry explains why the ellipse is formed."
        ]
        self.setup_layout("The Mathematical Magic: Proving the Ellipse", lecture_lines)
        
        # Initialize objects
        p_point = Dot(color=WHITE)
        self.place_at_grid(p_point, 'C3')
        
        dist_line1 = Line(start=self.grid['C3'], end=self.grid['B2'], color=YELLOW)
        dist_line2 = Line(start=self.grid['C3'], end=self.grid['B4'], color=YELLOW)
        
        ellipse = Ellipse(width=3, height=2, color=TEAL)
        self.place_in_area(ellipse, 'B1', 'E5')

        # Asset loading
        sphere_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        focus1 = self.place_at_grid(SVGMobject(sphere_path), 'B2', scale_factor=0.3)
        focus2 = self.place_at_grid(SVGMobject(sphere_path), 'B4', scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(p_point))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.play(Create(dist_line1), Create(dist_line2))
        self.lecture[1].set_color("#FFFF00")
        
        # === Animation for Lecture Line 3 ===
        self.play(Create(ellipse))
        self.lecture[2].set_color("#00FF00")
        
        # === Animation for Lecture Line 4 ===
        label1 = Text("F1", font_size=20, color=PURPLE)
        label2 = Text("F2", font_size=20, color=PURPLE)
        # Using manual positioning here based on the grid to ensure alignment
        label1.move_to(self.grid['B2'] + UP * 0.4)
        label2.move_to(self.grid['B4'] + UP * 0.4)
        self.play(FadeIn(focus1), FadeIn(focus2), Write(label1), Write(label2))
        self.lecture[3].set_color("#FF00FF")
        
        # === Animation for Lecture Line 5 ===
        self.play(Indicate(ellipse))
        self.lecture[4].set_color("#00FFFF")
        self.wait(2)
