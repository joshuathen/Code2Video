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
        lecture_lines = ["A circle's area is pi times radius squared.", "A sphere is points equidistant from center.", "Parallel projection creates a flat shadow."]
        self.setup_layout("Prerequisites: The Geometry of Circles and Spheres", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        circle = Circle(radius=1.0, color="#FF5733")
        self.place_in_area(circle, 'B2', 'B3', scale_factor=0.6)
        radius_line = Line(circle.get_center(), circle.get_right(), color=WHITE)
        radius_label = MathTex("r", color=WHITE).next_to(radius_line, UP, buff=0.1)
        area_formula = MathTex("A = \\pi r^2", color="#33FF57")
        self.place_at_grid(area_formula, 'C2', scale_factor=0.7)
        
        self.play(Create(circle), Write(radius_line), Write(radius_label))
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.play(Write(area_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#3357FF")
        self.place_in_area(sphere_icon, 'B5', 'B6', scale_factor=0.6)
        glow = Dot(color="#3357FF", radius=0.1).move_to(sphere_icon.get_center())
        
        self.play(FadeIn(sphere_icon), FadeIn(glow))
        self.play(self.lecture[1].animate.set_color("#3357FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        shadow = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color=GREY).scale(0.8)
        self.place_at_grid(shadow, 'C5', scale_factor=0.5)
        
        self.play(Create(shadow))
        self.play(self.lecture[2].animate.set_color(GREY))
        self.wait(2)
